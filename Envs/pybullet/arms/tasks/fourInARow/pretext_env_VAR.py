from Envs.pybullet.arms.tasks.fourInARow.fourInARow import FourInARow
try: import sounddevice as sd
except: pass
import gym
import numpy as np

class PretextEnvVAR(FourInARow):
	def __init__(self):
		FourInARow.__init__(self)

		# observation space
		d = {
			'image': gym.spaces.Box(low=0, high=255, shape=self.config.img_dim, dtype='uint8'),
			'ground_truth': gym.spaces.Box(low=0, high=self.config.taskNum + 1, shape=(1,), dtype=np.int32),
			'sound_negative_id': gym.spaces.Box(low=0, high=self.config.taskNum + 1, shape=(1,), dtype=np.int32),
		}

		if self.config.pretextDataHasSound:
			d['sound_positive'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.config.sound_dim, dtype=np.float32)
			d['sound_negative'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.config.sound_dim, dtype=np.float32)

		self.observation_space = gym.spaces.Dict(d)
		self.maxSteps = self.config.pretextEnvMaxSteps

		# setup action space
		high = np.ones(self.config.pretextActionDim)
		self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

	def gen_obs(self):
		self.image = self.robot.get_image(self.externalCamEyePosition, self.externalCamTargetPosition,
										  renderer=self.renderer)
		self.saveEpisodeImage(self.image)

		s = self.robot.calc_state()
		sound_positive, sound_negative, ground_truth, positive_audio, intent_negative = \
			self.get_positive_negative(generate_audio=self.config.pretextDataHasSound)

		if positive_audio is not None and self.config.render:
			sd.play(positive_audio, self.audio.fs)
		obs = {
			'image': np.transpose(self.image, (2, 0, 1)),  # for PyTorch convolution,
			'ground_truth': ground_truth,
			'sound_negative_id': intent_negative,
		}

		if self.config.pretextDataHasSound:
			obs['sound_positive'] = sound_positive
			obs['sound_negative'] = sound_negative

		return obs, s

	def callApplyAction(self, action):
		return self.robot.applyActionPretext(np.array(action), self.np_random)

	def callTestPolicy(self, infoDict):
		pass











