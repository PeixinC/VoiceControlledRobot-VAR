import numpy as np
try: import sounddevice as sd
except: pass
import gym
from .RL_env_VAR import RLEnvVAR, Task



class PretextEnvVAR(RLEnvVAR):
	def __init__(self):
		RLEnvVAR.__init__(self)
		d = {
			'image': gym.spaces.Box(low=0, high=255, shape=self.config.img_dim, dtype='uint8'),
			'sound_negative_id': gym.spaces.Box(low=0, high=self.config.taskNum+1, shape=(1,), dtype=np.int32),
			'ground_truth': gym.spaces.Box(low=0, high=self.config.taskNum+1, shape=(1,), dtype=np.int32), # sound positive label
		}

		if self.config.pretextDataHasSound:
			d['sound_positive'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.config.sound_dim, dtype=np.float32)
			d['sound_negative'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.config.sound_dim, dtype=np.float32)

		self.observation_space = gym.spaces.Dict(d)
		self.maxSteps = self.config.pretextEnvMaxSteps

		self.visibleDist = self.config.pretextVisibilityDistance

	def setupTask(self):
		self.domainRandomization()
		if self.task.act=='PickupObject':
			self.pickUpByTask(self.task)


	def get_pos_act(self, obj_in_view):
		if len(self.config.allTasks[self.task.loc][obj_in_view]) == 1:
				act = self.config.allTasks[self.task.loc][obj_in_view][0]
		else:
			# check the current state of the obj_in_view and choose the same
			# TODO: we only consider ToggleOn and ToggleOff for now
			if self.objMeta[obj_in_view]["isToggled"]:
				act = 'ToggleObjectOn'
			else:
				act = 'ToggleObjectOff'
		return act

	def get_positive_negative(self, get_negative, generate_audio):
		"""
		get positive and negative sound command
		:param get_negative: if generate negative sound
		:param generate_audio: audio is not needed for pretext envs (audios will be paired in dataset.py)
		:return: sound_positive, sound_negative, ground_truth, positive_audio, intent_negative
		"""
		sound_positive, sound_negative, positive_audio, intent_negative = None, None, None, None

		num_visible=0
		obj_in_view=None
		for k in self.visibility:
			if k != "Pillow":
				if self.visibility[k]:
					num_visible=num_visible+1
					obj_in_view=k

		inventory = self.controller.last_event.metadata['inventoryObjects']
		if len(inventory) != 0:
			pos_tsk = Task(loc=self.task.loc, obj=inventory[0]['objectType'], act='PickupObject')
			ground_truth = np.int32(self.task2ID[pos_tsk])
			if generate_audio or self.config.render:
				sound_positive, positive_audio, _ = self.audio.getAudioFromTask(self.np_random, pos_tsk, Task)
			if get_negative:
				intent_negative=self.get_negatives(empty=False, ground_truth=ground_truth)
				if generate_audio:
					if intent_negative == self.config.taskNum:
						sound_negative = np.zeros(shape=self.config.sound_dim)
					else:
						neg_tsk=self.taskList[intent_negative]
						sound_negative, negaitve_audio, _ = self.audio.getAudioFromTask(self.np_random, neg_tsk, Task)


		else:
			if num_visible!=1:
				# the agent sees nothing, no sound is given
				pos_tsk=None
				ground_truth = np.int32(self.config.taskNum)
				if generate_audio:
					sound_positive = np.zeros(shape=self.config.sound_dim)
				if get_negative:
					intent_negative=self.get_negatives(empty=True, ground_truth=ground_truth)
					if generate_audio:
						neg_tsk = self.taskList[intent_negative]
						sound_negative, negaitve_audio, _ = self.audio.getAudioFromTask(self.np_random, neg_tsk, Task)


			else:  # the agent sees an object
				act=self.get_pos_act(obj_in_view)

				pos_tsk = Task(loc=self.task.loc, obj=obj_in_view, act=act)
				ground_truth = np.int32(self.task2ID[pos_tsk])
				if generate_audio or self.config.render:
					sound_positive, positive_audio,_ = self.audio.getAudioFromTask(self.np_random, pos_tsk, Task)
				if get_negative:
					intent_negative=self.get_negatives(empty=False, ground_truth=ground_truth)
					if generate_audio:
						if intent_negative == self.config.taskNum:
							sound_negative = np.zeros(shape=self.config.sound_dim)
						else:
							neg_tsk = self.taskList[intent_negative]
							sound_negative, negaitve_audio, _ = self.audio.getAudioFromTask(self.np_random, neg_tsk, Task)

		return sound_positive, sound_negative, ground_truth, positive_audio, intent_negative

	def gen_obs(self):
		"""
		generate observation for the robot
		:return: a dict containing various type of observations
		"""
		# update object metadata
		self.updateObjMeta(list(self.objMeta.keys()))
		self.checkVisible()

		if self.config.render:
			self.agentMeta = self.controller.last_event.metadata["agent"]
			self.local_occupancy = self.get_local_occupancy_map(x=self.agentMeta['position']['x'],
																z=self.agentMeta['position']['z'],
																y=self.agentMeta['rotation']['y'])

		image=self.controller.last_event.frame
		self.saveEpisodeImage(image)

		sound_positive, sound_negative, ground_truth, positive_audio, intent_negative = \
			self.get_positive_negative(get_negative=True, generate_audio=self.config.pretextDataHasSound)
		if self.config.render and positive_audio is not None:
			sd.play(positive_audio, self.audio.fs)

		obs = {
			'image': np.transpose(image, (2, 0, 1)), # for PyTorch convolution,
			'sound_negative_id': np.array([intent_negative]),
			'ground_truth': np.array([ground_truth]),
		}

		if self.config.pretextDataHasSound:
			obs['sound_positive'] = sound_positive
			obs['sound_negative'] = sound_negative

		return obs, 0



	def step(self, action):
		action=np.array(action)
		infoDict = {}
		k=None

		if self.config.pretextManualControl or self.config.pretextManualCollect:
			k=self.keyboardControl()
		else:
			self.randomTeleport()

		self.controller.step("Pass") # fix the design choice that images from the Unity window lag by 1 step

		# update counters
		self.envStepCounter = self.envStepCounter + 1
		# get new obs
		obs,_ = self.gen_obs()

		if self.config.use3rdCam:
			self.update3rdCam("Update")


		if k == 'r':  # save this pair to buffer
			self.saved_pairs.append(obs)
			print("Number of pairs collected", len(self.saved_pairs))
		elif k == 'z':  # save collected pairs in the buffer to disk
			self.saveManualPairs()
			print("Data saved to", self.config.pretextDataDir[0])



		r =self.rewards() # calculate reward
		self.reward = sum(r)
		self.episodeReward = self.episodeReward + self.reward
		self.done = self.termination()

		return obs, self.reward, self.done, infoDict # reset will be called if done

