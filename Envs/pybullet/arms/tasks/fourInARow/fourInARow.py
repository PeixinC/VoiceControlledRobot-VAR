from Envs.pybullet.arms.env_bases import BaseEnv
from .kuka.robot_manipulators import Kuka
import cv2
try: import sounddevice as sd
except:
	print("Package sounddevice import failure")
	sd=None
from cfg import main_config
from Envs.audioLoader import audioLoader
import os
import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle
from datetime import datetime


class FourInARow(BaseEnv):
	"""
	The base class for 4 objects in a row on a table
	Currently supports Kuka
	"""
	def __init__(self):
		self.config=main_config()
		self.audio = None
		self.key_code = None # track what is pressed on the keyboard

		# setup
		if self.config.robotType=='kuka':
			self.robot =Kuka(self.config)
		else: raise NotImplementedError

		self.robotID = None
		self.image=np.zeros([self.config.img_dim[1],self.config.img_dim[2], 3], dtype=np.uint8)

		d = {
			'image': gym.spaces.Box(low=0, high=255, shape=self.config.img_dim, dtype='uint8'),
			'goal_sound': gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.config.sound_dim, dtype=np.float32),
			'current_sound': gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.config.sound_dim, dtype=np.float32),
			'robot_pose': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.robotStateDim,), dtype=np.float32),
			# [0:'left most block', 1:'left block', 2:'right block', 3:'right most block', 4:empty]
			'goal_sound_label': gym.spaces.Box(low=0, high=self.config.taskNum+1, shape=(1,), dtype=np.int32),
			'goal_sound_feat': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.representationDim,),
											  dtype=np.float32),
			'image_feat': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.representationDim,),
										 dtype=np.float32),

		}
		self.observation_space = gym.spaces.Dict(d)
		self.maxSteps = self.config.RLEnvMaxSteps

		# setup action space
		if self.config.continuousControl:
			high = np.ones(self.config.RLActionDim)
			self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
		else:
			self.action_space = gym.spaces.Discrete(len(self.config.allActions))

		BaseEnv.__init__(self, config=self.config,
						 render=self.config.render,
						 action_space=self.action_space,
						 observation_space=self.observation_space)

		# models
		self.objList = self.config.objList
		self.objUidList = []  # PyBullet Uid for each object. Same order as config.objList
		self.objOrder = None # the order of the objects. {objList index: object position index on the table}


		# each spot associates with a list [x,y,yaw]. The order will be same with self.objList
		# [[x,y,yaw],[x,y,yaw],...]
		self.objPoseList = []  # objects position and orientation information
		self.table_path=None
		self.tableUid = None
		self.baseUid=None
		self.base2Uid=None
		self.wallID=None
		self.texPath = os.path.join(self.config.commonMediaPath, 'texture_subset')
		self.textureList = []
		self.workSpaceDebugLine = []
		self.workSpaceGridLine=[]
		self.eePositionDebugLine = None
		self.hideObjIdx = []
		self.intentIdx = None
		self.goal_sound = None
		self.ground_truth = None # the ground truth label for the goal sound
		self.goal_audio = None
		self.rayTestResult=None
		self.externalCamEyePosition=self.config.externalCamEyePosition
		self.externalCamTargetPosition=self.config.externalCamTargetPosition
		self.saved_pairs = []

		self.size_per_class = np.zeros((self.config.taskNum,), dtype=np.int64)
		for key in self.config.soundSource['size']:
			self.size_per_class = self.size_per_class + self.config.soundSource['size'][key]
		self.size_per_class_cumsum = np.cumsum(self.size_per_class)

		self.goal_area_count=0


	def saveEpisodeImage(self, image):
		if self.config.episodeImgSaveInterval > 0 and self.episodeCounter % self.config.episodeImgSaveInterval == 0:
			# save the images
			imgSave = cv2.resize(image, (self.config.episodeImgSize[1], self.config.episodeImgSize[0]))
			if self.config.episodeImgSize[2] == 3:
				imgSave = cv2.cvtColor(imgSave, cv2.COLOR_RGB2BGR)
			fileName = str(self.episodeCounter) + '_' + str(self.envStepCounter) + '.jpg'
			cv2.imwrite(os.path.join(self.config.episodeImgSaveDir, fileName), imgSave)


	def loadTex(self):
		texList = os.listdir(self.texPath)
		idx = np.arange(len(texList))
		self.np_random.shuffle(idx)

		# load texture for walls
		for i in range(len(texList)):
			# key=fileName, val=textureID. If we have already loaded the texture, no need to reload and drain the memory
			texID = self._p.loadTexture(os.path.join(self.texPath, texList[idx[i]]))
			self.textureList.append(texID)
			if i >=self.config.numTexture:
				break

	def rand_rgb(self, val):
		"""
		:param val:
		:return: a list that contains r, g ,b values which satisfies following condition
				r + g + b >= 2 and r <= 1 and g <= 1 and b <= 1
		"""
		# generate 3 numbers in [0,1)
		rgb = self.np_random.rand(3)
		rgb = (1 - val) * rgb + val
		return rgb

	def changeWallTexture(self, wallID):
		texID = self.np_random.choice(self.textureList)
		r, g, b = self.rand_rgb(2. / 3)  # generate random rgb values
		self._p.changeVisualShape(wallID, -1, textureUniqueId=texID,
								  rgbaColor=[r, g, b, 1])

	def randomization(self):
		# randomize the locations of the objects
		randomx = self.np_random.uniform(self.config.xMin+self.config.objXRand[0], self.config.xMax+self.config.objXRand[1])
		randomy = self.np_random.uniform(self.config.yMin+self.config.objYRand[0], self.config.yMax+self.config.objYRand[1])


		if self.config.hideObj['mode']=='none':
			shuffled=np.arange(len(self.objList))
			self.np_random.shuffle(shuffled)
			self.objOrder=dict(zip(range(len(self.objList)), shuffled))

			for i in range(len(self.objList)):
				orn = self._p.getQuaternionFromEuler([0, 0, 0])
				y = randomy + self.objOrder[i] * self.config.objInterval + self.np_random.uniform(
					self.config.objsYRand[0], self.config.objsYRand[1])
				x = randomx + self.np_random.uniform(self.config.objsXRand[0], self.config.objsXRand[1])
				self._p.resetBasePositionAndOrientation(self.objUidList[i],
														[x, y, self.config.objZ[self.objList[i]]],
														orn)
		else:
			raise NotImplementedError


		eePositionX=self.np_random.uniform(self.config.xMin+self.config.eeXInitRand[0], self.config.xMax+self.config.eeXInitRand[1])
		eePositionY=self.np_random.uniform(self.config.yMin+self.config.eeYInitRand[0], self.config.yMax+self.config.eeYInitRand[1])

		self.robot.robot_specific_reset(eePositionX, eePositionY, self.config.endEffectorHeight)

		for _ in range(20):
			self._p.stepSimulation() # refresh the simulator. Needed for the ray test

	def get_positive_negative(self, get_negative=True, generate_audio=True):
		rayTest = self.robot.ray_test(self.objUidList)
		contactRays = [True if rayTest == Uid else False for Uid in self.objUidList]
		sound_positive, sound_negative,positive_audio, intent_negative=None, None, None, None

		if not any(contactRays):  # the end effector hits nothing, no sound is given
			intent_positive=self.config.taskNum # empty

			if generate_audio:
				sound_positive = np.zeros(shape=self.config.sound_dim)

			if get_negative:
				intent_negative = self.np_random.randint(0, self.config.taskNum)
				if generate_audio:
					sound_negative, _ = self.audio.genSoundFeat(intentIdx=intent_negative, featType='MFCC',
																rand_fn=self.np_random.randint)

		else:  # the agent sees an object in self.config.objList
			# decide intent_positive
			if self.config.commandType=='order':
				intent_positive = self.objOrder[np.argmax(contactRays)]
			else: raise NotImplementedError

			if generate_audio or self.config.render:
				sound_positive, positive_audio = self.audio.genSoundFeat(intentIdx=intent_positive, featType='MFCC',
																		 rand_fn=self.np_random.randint)
			if get_negative:
				intent_negative = self.np_random.randint(0, self.config.taskNum)
				if intent_positive == intent_negative:
					intent_negative=self.config.taskNum
					if generate_audio: sound_negative = np.zeros(shape=self.config.sound_dim)
				else:
					if generate_audio:
						sound_negative, negative_audio = self.audio.genSoundFeat(intentIdx=intent_negative, featType='MFCC',
																				 rand_fn=self.np_random.randint)

		ground_truth = np.int32(intent_positive)
		return sound_positive, sound_negative, ground_truth, positive_audio, intent_negative

	def envReset(self):
		if self.robot.robot_ids is None:
			# load sound
			if self.audio is None:
				self.audio=audioLoader(config=self.config)
			# load robot
			self.robot.load_model()
			self.robot._p = self._p
			self.robotID = self.robot.robot_ids

			# anchor the robot
			self._p.resetBasePositionAndOrientation(self.robotID, self.config.robotPosition, [0.0, 0.0, 0.0, 1.0])

			# load table and obj
			self.robot.setup_scene(self)

			self.tableUid = self._p.loadURDF(self.table_path,
											 self.config.tablePosition,
											 [0.0, 0.0, 0.0, 1.0])
			self.robot.tableUid = self.tableUid

			for i in range(len(self.objList)):
				objPath = os.path.join(self.config.mediaPath, 'objects', 'fourInARow', self.config.objList[i], self.config.objList[i]+'.urdf')
				newUid = self._p.loadURDF(objPath)
				self.objUidList.append(newUid)

			if self.config.render:
				self.drawRectangleDebug(self.workSpaceDebugLine, self.config.xMin, self.config.xMax, self.config.yMin, self.config.yMax, 0)
				if not self.config.continuousControl:
					self.drawGrid(self.workSpaceGridLine, self.config.discretizeGridSize, self.config.xMin,
								  self.config.xMax, self.config.yMin, self.config.yMax, 0)


			self.robot._p = self._p
			self.randomization()

		if self.config.ifReset and self.episodeCounter > 0:
			self.randomization()

		self.goal_area_count = 0
		ret = self.gen_obs()
		return ret[0]

	def getIntentIdx(self):
		if self.config.hideObj['mode'] == 'none':
			# randomly select an object
			if self.config.RLTrain or self.config.render:
				self.intentIdx = self.np_random.randint(0, self.config.taskNum)

			else:
				idx = np.where(self.size_per_class_cumsum <= self.episodeCounter)[0]
				self.intentIdx = 0 if len(idx) == 0 else min(int(idx.max() + 1), self.config.taskNum - 1)
		else:
			raise NotImplementedError


	def getManualIntent(self, pretext=False):
		while True:
			x = int(input("Please input the intent ID "))
			upperBound=self.config.taskNum+1 if pretext else self.config.taskNum # in pretext, taskNum+1 is Empty intent
			if 0 <= int(x) < upperBound:
				return x
			else:
				print("Invalid intent ID")


	def setupFirstStep(self):
		"""
		procedure for the first step of an episode, used for RL env
		"""
		self.getIntentIdx()
		self.goal_sound, self.goal_audio = self.audio.genSoundFeat(intentIdx=self.intentIdx, featType='MFCC',
																   rand_fn=self.np_random.randint)

		self.ground_truth = np.int32(self.intentIdx)
		if self.config.render or self.config.RLTrain == False:
			if self.goal_audio is not None and self.config.render:
				if sd:
					sd.play(self.goal_audio, self.audio.fs)

			print('Goal intent is', self.intentIdx)


	def gen_obs(self):
		self.image= self.robot.get_image(self.externalCamEyePosition, self.externalCamTargetPosition, renderer=self.renderer)
		self.saveEpisodeImage(self.image)
		s = self.robot.calc_state()


		if self.envStepCounter==0:
			self.setupFirstStep()

		sound_positive, sound_negative, sound_positive_ground_truth, positive_audio, _ = self.get_positive_negative(get_negative=False)

		obs = {
			'image': np.transpose(self.image, (2, 0, 1)),
			'goal_sound': self.goal_sound,
			'current_sound': sound_positive,
			'robot_pose': np.array([s['eeState'][0], s['eeState'][1]]),
			'goal_sound_label': self.ground_truth,
			'goal_sound_feat': np.zeros((self.config.representationDim,)),
			'image_feat': np.zeros((self.config.representationDim,))
		}

		return obs, s

	def testPolicy(self, infoDict):
		"""
		The agent succeeds if its gripper is above the object at the last timestep
		:param infoDict:
		:return:
		"""
		rayTest = self.robot.ray_test(self.objUidList)
		contactRays = [True if rayTest == Uid else False for Uid in self.objUidList]

		if self.done and any(contactRays):
			if self.config.commandType == 'order':
				if self.objOrder[np.argmax(contactRays)] == self.intentIdx:
					self.goal_area_count = self.goal_area_count + 1
			else:
				raise NotImplementedError

		if self.done:
			infoDict['goal_area_count'] = self.goal_area_count
			print('goal area count', self.goal_area_count)

	def saveManualPairs(self):
		"""
		used to save data pair when doing manual collection
		"""
		filePath = os.path.join(self.config.pretextDataDir[0], 'train')
		if not os.path.isdir(filePath):
			os.makedirs(filePath)

		datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
		filePath = os.path.join(filePath, 'data_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.pickle')

		with open(filePath, 'wb') as f:
			pickle.dump(self.saved_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
		self.saved_pairs.clear()
		print("Data saved to", self.config.pretextDataDir[0])

	def callApplyAction(self, action):
		return self.robot.applyAction(np.array(action))

	def callTestPolicy(self, infoDict):
		# at test time, perform rayTest and put performance into the infoDict
		if not self.config.RLTrain:
			self.testPolicy(infoDict)

	def step(self, action):
		self.key_code = self.callApplyAction(action)
		self.scene.global_step()
		self.envStepCounter = self.envStepCounter + 1

		obs, s= self.gen_obs()

		infoDict = {}

		if self.key_code == 'r':  # save this pair to buffer
			self.saved_pairs.append(obs)
			if 'ground_truth' in obs:
				print("Number of pairs collected", len(self.saved_pairs), 'ground_truth is', obs['ground_truth'])
			else:
				print("Number of pairs collected", len(self.saved_pairs), 'ground_truth is', obs['goal_sound_label'])
		elif self.key_code == 'z':  # save collected pairs in the buffer to disk
			self.saveManualPairs()


		r = [self.rewards()]  # calculate reward
		self.reward = sum(r)
		self.episodeReward = self.episodeReward + self.reward
		self.done = self.termination(s)


		self.callTestPolicy(infoDict)


		return obs, self.reward, self.done, infoDict  # reset will be called if done



	def termination(self, s):
		if self.envStepCounter >= self.maxSteps:
			return True
		return False

	def rewards(self, *args):
		reward=0.
		if self.config.RLUseEnvReward:
			rayTest = self.robot.ray_test(self.objUidList)
			contactRays = [True if rayTest == Uid else False for Uid in self.objUidList]
			if any(contactRays) and self.objOrder[np.argmax(contactRays)] == self.intentIdx:
				reward= 1.

		return reward