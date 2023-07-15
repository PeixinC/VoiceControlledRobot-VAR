import pandas as pd
import torch
import numpy as np
import os
from cfg import main_config
from RL import RLBase

class RL_VAR(RLBase):
    def __init__(self):
        super().__init__(main_config())
        
    def testRL(self, eval_envs):

        baseEnv = eval_envs.venv.unwrapped.envs[0]

        # load the trained policy
        skillList=self.loadPolicy(eval_envs)

        eval_episode_rewards = []
        eval_env_rewards = 0.

        obs = eval_envs.reset()

        eval_recurrent_hidden_states = torch.zeros(
            1, skillList[0].recurrent_hidden_state_size, device=self.device)
        eval_masks = torch.zeros(1, 1, device=self.device)

        episode_num = baseEnv.size_per_class_cumsum[-1]

        results = []
        goal_area_count_list = []
        objs = np.arange(self.config.taskNum, dtype=np.int64)
        objs = np.repeat(objs, baseEnv.size_per_class)

        while baseEnv.episodeCounter < episode_num:

            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = skillList[0].act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=self.config.RLDeterministic)

            # Obser reward and next obs
            obs, _, done, infos = eval_envs.step(action)
            if self.config.render:
                eval_envs.render()

                print('step reward', eval_envs.venv.origStepReward)
            eval_env_rewards = eval_env_rewards + eval_envs.venv.origStepReward

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=self.device)

            if done:
                goal_area_count = infos[0]['goal_area_count']
                goal_area_count_list.append(goal_area_count)
                results.append(int(goal_area_count >= self.config.success_threshold))
                eval_episode_rewards.append(float(eval_env_rewards))
                eval_env_rewards = 0.

        # save the results
        if not self.config.render:
            df = pd.DataFrame(
                {'objIdx': objs, 'goal area count': goal_area_count_list, 'rewards': eval_episode_rewards,
                 'results': results})

            save_path = os.path.join(os.path.dirname(self.config.skillInfos[0]['path']), 'test_' +
                                     os.path.splitext(os.path.basename(self.config.skillInfos[0]['path']))[
                                         0] + '.csv')
            df.to_csv(save_path, mode='w', header=True, index=False)
            print('results saved to', save_path)
            print('success rate', sum(results) * 1. / baseEnv.size_per_class_cumsum[-1])
        eval_envs.close()
