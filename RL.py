import torch
from cfg import gym_register, main_config, TASK, ENV
import os
import pandas as pd
import time
import numpy as np
from shutil import copyfile
import torch.backends.cudnn as cudnn
from Envs.vec_env.envs import make_vec_envs
from models.ppo.model import Policy
from models.ppo.storage import RolloutStorage
from models.ppo import algo
from collections import deque
import gym
from pretext import Pretext


class RLBase(object):
    def __init__(self, config):
        self.config = config
        gym_register(self.config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.pretextObj = Pretext(self.config)


    def manualControl(self, envs):

        observation = envs.reset()
        for episode in range(50):

            for i in range(self.config.RLEnvMaxSteps):
                print('step:', i)
                print('step reward', envs.venv.origStepReward)
                envs.render()
                time.sleep(0.1)
                action = torch.zeros(self.config.RLActionDim)  # dummy action. True action is decided in env
                observation, _, _, _ = envs.step(action)

    def loadPolicy(self, envs):
        skillList = []
        for i, skill_info in enumerate(self.config.skillInfos):
            assert os.path.exists(skill_info['path'])
            if envs.action_space.__class__.__name__ == "Discrete":
                action_space = gym.spaces.Discrete(skill_info['actionDim'])
            elif envs.action_space.__class__.__name__ == "Box":
                high = np.ones(skill_info['actionDim'])
                action_space = gym.spaces.Box(-high, high, dtype=np.float32)
            else:
                raise NotImplementedError
            ac = Policy(
                envs.venv.observation_space.spaces,
                action_space,
                base=self.config.RLPolicyBase,
                config=self.config,
                base_kwargs={'recurrent': self.config.RLRecurrentPolicy,
                             'recurrentInputSize': self.config.RLRecurrentInputSize,
                             'recurrentSize': self.config.RLRecurrentSize,
                             'actionHiddenSize': self.config.RLActionHiddenSize
                             })
            print("Load the weights from", skill_info['path'])
            ac.load_state_dict(torch.load(skill_info['path']))
            ac.eval()
            print("Weights Loaded!")
            ac.to(self.device)
            skillList.append(ac)

        skillNum=len(skillList)
        assert skillNum!= 0

        return skillList

                
    def trainRL(self):
        torch.set_num_threads(1)
        torch.manual_seed(self.config.RLEnvSeed)
        torch.cuda.manual_seed_all(self.config.RLEnvSeed)

        if not os.path.exists(self.config.RLModelSaveDir):
            os.makedirs(self.config.RLModelSaveDir)

        if ENV == 'arms':
            copyfile(os.path.join('Envs', self.config.envFolder, 'tasks', TASK, 'config.py'),
                     os.path.join(self.config.RLModelSaveDir, 'config.py'))
        else:
            copyfile(os.path.join('Envs', self.config.envFolder, 'config.py'),
                     os.path.join(self.config.RLModelSaveDir, 'config.py'))

        envs = make_vec_envs(env_name=self.config.RLEnvName,
                             seed=self.config.RLEnvSeed,
                             num_processes=self.config.RLNumEnvs,
                             gamma=self.config.RLGamma,
                             device=self.device,
                             randomCollect=False,
                             config=self.config,
                             pretextObj=self.pretextObj
                             )

        actor_critic = Policy(
            envs.venv.observation_space.spaces,
            envs.action_space,
            config = self.config,
            base = self.config.RLPolicyBase,
            base_kwargs = {'recurrent': self.config.RLRecurrentPolicy,
                        'recurrentInputSize': self.config.RLRecurrentInputSize,
                        'recurrentSize': self.config.RLRecurrentSize,
                        'actionHiddenSize': self.config.RLActionHiddenSize
                        })
        actor_critic.to(self.device)

        if self.config.RLModelFineTune:
            print("Load the weights from", self.config.RLModelLoadDir)
            actor_critic.load_state_dict(torch.load(self.config.RLModelLoadDir))

        agent = algo.PPO(
            actor_critic,
            self.config.ppoClipParam,
            self.config.ppoEpoch,
            self.config.ppoNumMiniBatch,
            self.config.ppoValueLossCoef,
            self.config.ppoEntropyCoef,
            lr=self.config.RLLr,
            eps=self.config.RLEps,
            max_grad_norm=self.config.RLMaxGradNorm,
            config = self.config)

        rollouts = RolloutStorage(self.config.ppoNumSteps, self.config.RLNumEnvs,
                                  envs.venv.observation_space.spaces, envs.action_space,
                                  actor_critic.recurrent_hidden_state_size, config = self.config)

        env_rewards = np.zeros([self.config.RLNumEnvs, ])
        episode_rewards = deque(maxlen=10)

        print('Begin RL training')
        obs = envs.reset()

        if isinstance(rollouts.obs, dict):
            for key in rollouts.obs:
                rollouts.obs[key][0].copy_(obs[key])
        else:
            rollouts.obs[0].copy_(obs)
        rollouts.to(self.device)

        start = time.time()
        num_updates = int(
            self.config.RLTotalSteps) // self.config.ppoNumSteps // self.config.RLNumEnvs
        for j in range(0, num_updates):
            for step in range(self.config.ppoNumSteps):
                # Sample actions
                with torch.no_grad():
                    if isinstance(rollouts.obs, dict):
                        rollouts_obs = {}
                        for key in rollouts.obs:
                            rollouts_obs[key] = rollouts.obs[key][step]
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                            rollouts_obs, rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step])
                    else:
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                            rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step])

                # Obs reward and next obs
                obs, reward, done, infos = envs.step(action)

                if self.config.render:
                    print('step reward', envs.venv.origStepReward)

                    envs.render()

                env_rewards = env_rewards + envs.venv.origStepReward
                if any(done):
                    idx = np.where(done == True)[0]
                    for index in idx:
                        episode_rewards.append(env_rewards[index])
                        env_rewards[index] = 0.

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                if isinstance(rollouts.obs, dict):
                    rollouts_obs = {}
                    for key in rollouts.obs:
                        rollouts_obs[key] = rollouts.obs[key][-1]
                    next_value = actor_critic.get_value(
                        rollouts_obs, rollouts.recurrent_hidden_states[-1],
                        rollouts.masks[-1]).detach()
                else:
                    next_value = actor_critic.get_value(
                        rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                        rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, self.config.ppoUseGAE, self.config.RLGamma,
                                     self.config.ppoGAELambda,
                                     self.config.RLUseProperTimeLimits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            # save for every interval-th episode or for the last epoch
            if (j % self.config.RLModelSaveInterval == 0
                or j == num_updates - 1) and self.config.RLModelSaveDir != "":
                save_path = self.config.RLModelSaveDir

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(actor_critic.state_dict(), os.path.join(save_path, '%.5i' % j + ".pt"),
                           _use_new_zipfile_serialization=False)

            if j % self.config.RLLogInterval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * self.config.RLNumEnvs * self.config.ppoNumSteps
                end = time.time()
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                        .format(j, total_num_steps,
                                int(total_num_steps / (end - start)),
                                len(episode_rewards), np.mean(episode_rewards),
                                np.median(episode_rewards), np.min(episode_rewards),
                                np.max(episode_rewards), dist_entropy, value_loss,
                                action_loss))

                df = pd.DataFrame({'misc/nupdates': [j], 'misc/total_timesteps': [total_num_steps],
                                   'fps': int(total_num_steps / (end - start)),
                                   'eprewmean': [np.mean(episode_rewards)],
                                   'min': np.min(episode_rewards),
                                   'max': np.max(episode_rewards),
                                   'loss/policy_entropy': dist_entropy, 'loss/policy_loss': action_loss,
                                   'loss/value_loss': value_loss})

                if os.path.exists(os.path.join(self.config.RLModelSaveDir, 'progress.csv')) and j > 20:
                    df.to_csv(os.path.join(self.config.RLModelSaveDir, 'progress.csv'), mode='a', header=False,
                              index=False)
                else:
                    df.to_csv(os.path.join(self.config.RLModelSaveDir, 'progress.csv'), mode='w', header=True,
                              index=False)

        envs.close()


    def testRL(self, eval_envs):
        raise NotImplementedError("Please Implement this method")

    def run(self):
        cudnn.benchmark = True
        torch.cuda.empty_cache()


        if not (self.config.RLManualControl and not self.config.RLManualControlLoaded):
            self.pretextObj.loadPretextModel()


        if self.config.RLManualControl:
            envs = make_vec_envs(env_name=self.config.RLEnvName,
                                 seed=self.config.RLEnvSeed,
                                 num_processes=1,
                                 gamma=self.config.RLGamma,
                                 device=self.device,
                                 randomCollect=False,
                                 config=self.config,
                                 pretextObj=self.pretextObj)
            self.manualControl(envs)
        else:

            if self.config.RLTrain:
                self.trainRL()

            else:  # evaluate the policy
                envs = make_vec_envs(env_name=self.config.RLEnvName,
                                     seed=self.config.RLEnvSeed,
                                     num_processes=1,
                                     gamma=self.config.RLGamma,
                                     device=self.device,
                                     randomCollect=False,
                                     config=self.config,
                                     pretextObj=self.pretextObj)
                self.testRL(envs)

if __name__ == '__main__':
    mc=main_config()
    from VAR.RL_VAR import RL_VAR

    rl_var=RL_VAR()
    rl_var.run()