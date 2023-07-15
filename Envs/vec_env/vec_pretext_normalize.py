from . import VecEnvWrapper
import numpy as np
from .running_mean_std import RunningMeanStd
import torch
from utils import drawArrows


class VecPretextNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, config=None, pretextObj=None):
        VecEnvWrapper.__init__(self, venv)

        self.config = config
        self.pretextObj=pretextObj
        self.pretextModel=self.pretextObj.pretextModel
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.medoids=None
        self.new_sound_feat=None
        self.fig=None
        self.ax=None
        self.figText=None
        self.fileNum=0
        self.quiver_img=None
        self.quiver_sound =None

        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

        self.origStepReward = np.zeros(self.num_envs)
        self.rl_obs_space=None

        self.processing_func={
            'ArmConfig': self.processArm, 'AI2ThorConfig':self.processAI2Thor
        }


    def step_wait(self):
        obs, env_rews, news, infos = self.venv.step_wait()
        # process the observations and reward
        obs,rews=self.processing_func[self.config.name](obs, env_rews, news, infos)


        self.origStepReward=rews.copy()
        # normalize the reward
        self.ret = self.ret * self.gamma + rews
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.

        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms and self.config.RLTrain:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        if self.config.realTimeVec:
            self.fig, self.ax, self.figText=self.pretextObj.initRealTimePlot()

        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        obs, _ = self.processing_func[self.config.name](obs, np.zeros((self.num_envs,)), np.array([True]*self.num_envs), ({},)*self.num_envs)
        

        return obs

    def getEmbeddings(self, O):
        with torch.no_grad():
            current_sound=torch.from_numpy(O['current_sound']).float().to(self.device) if self.config.RLRewardSoundSound else None
            d = self.pretextModel(torch.from_numpy(O['image'] / 255.).float().to(self.device),
                                  torch.from_numpy(O['goal_sound']).float().to(self.device),
                                  current_sound)
            image_feat = d['image_feat'].to('cpu').numpy()
            goal_sound_feat = d['sound_feat_positive'].to('cpu').numpy()
            if self.config.RLRewardSoundSound:
                current_sound_feat = d['sound_feat_negative'].to('cpu').numpy()
            else:
                current_sound_feat=0.
            return image_feat, goal_sound_feat, current_sound_feat

    def calcReward(self, envReward, image_feat=None, goal_sound_feat=None, current_sound_feat=None):
        img_sound_dot = np.sum(image_feat[:, :self.config.representationDim] * goal_sound_feat, axis=1)
        sound_sound_dot = np.sum(current_sound_feat * goal_sound_feat, axis=1)
        embReward = img_sound_dot + sound_sound_dot * self.config.RLRewardSoundSound
        reward = embReward + envReward
        return reward, img_sound_dot, sound_sound_dot


    def processArm(self, O, envReward, done, infos):
        if self.pretextModel is None: # no pretextModel
            return O, envReward

        image_feat, goal_sound_feat, current_sound_feat=self.getEmbeddings(O)

        reward, _, _=self.calcReward(envReward, image_feat, goal_sound_feat, current_sound_feat)
        s = {'robot_pose': O['robot_pose'], 'goal_sound_feat': goal_sound_feat,
             'image': O['image'] / 255.,
             'image_feat': image_feat,
             }

        if self.config.realTimeVec:
            self.quiver_img, self.quiver_sound = drawArrows(self.ax, self.fig, v_img=image_feat,
                                                            v_sound=goal_sound_feat, quiver_img=self.quiver_img,
                                                            quiver_sound=self.quiver_sound)

        obs = self._obfilt(s)

        return obs, reward

    def processAI2Thor(self, O, envReward, done, infos):
        if self.pretextModel is None: # no pretextModel
            return O, envReward

        image_feat, goal_sound_feat, current_sound_feat=self.getEmbeddings(O)


        reward, _, _=self.calcReward(envReward, image_feat, goal_sound_feat, current_sound_feat)
        s = {'occupancy': O['occupancy'] / 255.,
             'goal_sound_feat': goal_sound_feat,
             'image': O['image'] / 255.,
             'image_feat': image_feat,
             }

        if self.config.realTimeVec:
            self.quiver_img, self.quiver_sound = drawArrows(self.ax, self.fig, v_img=image_feat,
                                                            v_sound=goal_sound_feat, quiver_img=self.quiver_img,
                                                            quiver_sound=self.quiver_sound)

        obs = self._obfilt(s)

        return obs, reward