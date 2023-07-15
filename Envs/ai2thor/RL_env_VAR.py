import os
import sys
from sys import platform
if platform == "linux" or platform == "linux2":
	import tty
	import termios
import numpy as np
import cv2
try: import sounddevice as sd
except: pass
import gym
from ai2thor.controller import Controller
import matplotlib.pyplot as plt
from cfg import main_config
from Envs.audioLoader import audioLoader
import time
from datetime import datetime
import pickle
import warnings
from scipy import ndimage
from ai2thor.platform import CloudRendering

class Task(object):
	def __init__(self, loc, obj, act):
		self.loc=loc
		self.obj=obj
		self.act=act
	def __eq__(self, other):
		return (self.loc, self.obj, self.act)==(other.loc, other.obj, other.act)

	def __hash__(self):
		return hash((self.loc, self.obj, self.act))

	def __ne__(self, other):
		return not (self == other)

class RLEnvVAR(gym.Env):
	def __init__(self):

		self.config = main_config() # load env config
		self.audio = audioLoader(config=self.config)

		# observation space
		d = {
			'image': gym.spaces.Box(low=0, high=255, shape=self.config.img_dim, dtype='uint8'),
			'occupancy': gym.spaces.Box(low=0, high=255, shape=(1, self.config.RLVisibleGrid, self.config.RLVisibleGrid), dtype='uint8'),
			'goal_sound': gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.config.sound_dim, dtype=np.float32),
			'current_sound': gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.config.sound_dim, dtype=np.float32),
			'goal_sound_label': gym.spaces.Box(low=0, high=self.config.taskNum+1, shape=(1,), dtype=np.int32),
			'goal_sound_feat': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.representationDim,),
											  dtype=np.float32),
			'image_feat': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.representationDim,),
										 dtype=np.float32),
		}

		self.observation_space = gym.spaces.Dict(d)
		self.maxSteps = self.config.RLEnvMaxSteps

		# setup action space
		self.action_space = gym.spaces.Discrete(len(self.config.allActions))

		self.visibleDist = self.config.RLVisibilityDistance
		self.renderInstanceSegmentation = False


		#episode related
		self.episodeCounter=0
		self.envStepCounter=0
		self.done = 0
		self.reward = 0
		self.episodeReward = 0.0
		self.terminated = False
		self.saved_pairs=[]

		# random and seed
		self.np_random=None
		self.givenSeed=None

		# the ai2thor task
		self.controller=None # ai2thor controller
		self.taskID=None
		self.task=None # a Task object
		# taskList: a list of all possible tasks. The index in the list is the taskID
		# task2ID: a dict {Task():0, Task():1}
		self.taskList, self.taskLocRange, self.task2ID=self.parseTask()

		# variables that change every timestep or every episode
		self.floor_plan = None  # current floor plan index
		self.objMeta= {} # a dict containing object metadata relevant to the task. {key:objType, val:metadata}
		self.agentMeta=None # the metadata about the agent
		self.visibility = {}  # a dict containing visibility of objects. {key: objType, val: True/False}
		self.bb_dict={}

		# properties or information of each floor plan
		self.reachablePositions={} # robot's reachable positions for each floor plan
		self.occupancy_grid={} # a obstacle map for each floor plan
		self.min_xz={} # the minimum positions for each floor plan. {[min x, min z]}
		self.robotY={} # robot vertical position for each floor plan
		self.local_occupancy=None # local occupancy map



		# audio related
		self.audioPath={} # a nested dict holding dataframes location (outer dict key)->object->action (inner dict key)
		self.goal_sound = None
		self.goal_audio = None
		self.transcription=None

		# render
		if self.config.render:
			plt.ion()
			self.envFig, self.envAx=plt.subplots()
			self.envAxShow=self.envAx.imshow(np.zeros([self.config.img_dim[1],
													   self.config.img_dim[2],3], dtype=np.uint8))
			self.envAx.axis('off')
			self.envAxText=self.envAx.text(0.5, 0.02, "",
				transform=plt.gcf().transFigure, fontsize=14, ha='center', color='blue')
			self.envFig.canvas.draw_idle()
			self.envFig.canvas.start_event_loop(0.001)

			self.mapFig, self.mapAx=plt.subplots()
			self.mapAxShow = self.mapAx.imshow(np.zeros([self.config.RLVisibleGrid,self.config.RLVisibleGrid], dtype=np.uint8))

			self.mapFig.canvas.draw_idle()
			self.mapFig.canvas.start_event_loop(0.001)

			if self.config.use3rdCam:
				self.Fig3rd, self.Ax3rd = plt.subplots()
				self.AxShow3rd = self.Ax3rd.imshow(np.zeros([self.config.img_dim[1],
															 self.config.img_dim[2], 3], dtype=np.uint8))
				self.Ax3rd.axis('off')
				self.Fig3rd.canvas.draw_idle()
				self.Fig3rd.canvas.start_event_loop(0.001)



		self.size_per_class = np.zeros((self.config.taskNum,), dtype=np.int64)
		self.size_per_class = self.size_per_class + self.config.soundSource['size']
		self.size_per_class_cumsum = np.cumsum(self.size_per_class)

		self.goal_area_count = 0



	def parseTask(self):
		tl=[] # taskList
		tlr={} # taskLocRange
		t2I={} # task2ID

		for loc in self.config.allTasks:
			c = len(tl)
			for obj in self.config.allTasks[loc]:
				for act in self.config.allTasks[loc][obj]:
					t=Task(loc=loc, obj=obj, act=act)
					tl.append(t)
					t2I[t]=len(tl)-1
			tlr[loc] = [c, len(tl)]
		return tl, tlr, t2I


	def updateObjMeta(self, objTypes):

		for item in objTypes:
			for o in self.controller.last_event.metadata["objects"]:
				if o["objectType"]==item:
					self.objMeta[o["objectType"]]=o
					break

	def get_occupancy_grid(self, gridSize, max_xz):
		reachable_position_set = set(self.reachablePositions[self.floor_plan])

		padding=self.config.RLVisibleGrid+3
		min_xz=self.min_xz[self.floor_plan]=self.min_xz[self.floor_plan]-padding*self.config.gridSize[self.floor_plan]
		max_xz=max_xz+padding*self.config.gridSize[self.floor_plan]


		X, Z = np.meshgrid(np.arange(min_xz[0], max_xz[0] + gridSize, gridSize),
						   np.arange(min_xz[1], max_xz[1] + gridSize, gridSize))
		xz = np.array([X.flatten(), Z.flatten()]).T
		col_num = int((max_xz[0] - min_xz[0]) / gridSize + 1)
		row_num = int((max_xz[1] - min_xz[1]) / gridSize + 1)


		# 255 means occupied
		occupancy_grid = np.array(np.ones_like(xz[:, 0]) * 255, dtype=np.uint8).reshape((row_num, col_num))
		xz = np.reshape(xz, (row_num, col_num, 2))
		for i in range(row_num):
			for j in range(col_num):
				if tuple(xz[i][j]) in reachable_position_set:
					occupancy_grid[row_num - i - 1][j] = 0
		self.occupancy_grid[self.floor_plan]=occupancy_grid

	def get_local_occupancy_map(self, x, z, y):
		min_xz=self.min_xz[self.floor_plan]
		row_num, col_num=self.occupancy_grid[self.floor_plan].shape
		row_in_grid=int(row_num-(z-min_xz[1])/self.config.gridSize[self.floor_plan]-1)
		col_in_grid=int((x-min_xz[0])/self.config.gridSize[self.floor_plan])
		radius=self.config.RLVisibleGrid//2


		visible=self.occupancy_grid[self.floor_plan][row_in_grid-radius:row_in_grid+radius+1,
		   col_in_grid-radius:col_in_grid+radius+1]
		if visible.shape!=(self.config.RLVisibleGrid, self.config.RLVisibleGrid):
			print("Floor plan", self.floor_plan)
			print("x, y, z", x, y, z)
		rotated=ndimage.rotate(visible, y, reshape=False, order=0)
		rotated[radius, radius]=128

		return rotated


	def domainRandomization(self):

		if self.floor_plan==227:
			self.controller.step(
				action="ToggleObjectOff",
				objectId='LightSwitch|+00.00|+01.19|+04.41',
				forceAction=True
			)


		# domain randomization
		if 'InitialRandomSpawn' in self.config.domainRandomization:
			# random object location
			event=self.controller.step(action="InitialRandomSpawn", randomSeed=self.givenSeed,
								 forceVisible=False, numPlacementAttempts=5, placeStationary=True,
								 numDuplicatesOfType=[], excludedReceptacles=[], excludedObjectIds=[])
			if not event.metadata['lastActionSuccess']:
				warnings.warn(event.metadata['errorMessage'])

		if 'randomInitialPose' in self.config.domainRandomization:
			# random agent start pose
			self.randomTeleport()

		self.updateObjMeta(list(self.config.allTasks[self.task.loc].keys()))


		if 'randomObjState' in self.config.domainRandomization:
			for obj in ['FloorLamp', 'Television']:
				for o in self.controller.last_event.metadata["objects"]:
					if o["objectType"] == obj:
						objID=o["objectId"]
						event=self.controller.step(action=self.np_random.choice(["ToggleObjectOff", "ToggleObjectOn"]),
							objectId=objID, forceAction=True)

		self.updateObjMeta(self.config.allTasks[self.task.loc].keys())




	def setupTask(self):

		self.domainRandomization()

		if self.task.act=='ToggleObjectOn': # turn it off first
			event=self.controller.step(action="ToggleObjectOff",objectId=self.objMeta[self.task.obj]["objectId"],
								 forceAction=True)

		elif self.task.act=='ToggleObjectOff': # turn it on first
			event=self.controller.step(action="ToggleObjectOn",objectId=self.objMeta[self.task.obj]["objectId"],
								 forceAction=True)

		elif self.task.act=='PickupObject':
			pass
		else:
			raise NotImplementedError



	def checkVisible(self):
		for obj in self.config.allTasks[self.task.loc]:
			self.visibility[obj]=self.objMeta[obj]['visible']

	def reset(self):
		# choose the task
		# decide the types of rooms and the floor_plan
		self.taskID=self.np_random.randint(len(self.taskList))
		self.task=self.taskList[self.taskID]
		self.floor_plan = self.np_random.choice(self.config.allScene[self.task.loc])


		if self.controller is None: # if it is the first round

			iTHOR_platform=None if self.config.renderUnity else CloudRendering
			self.controller = Controller(agentMode="default", visibilityDistance=self.visibleDist, platform=iTHOR_platform,
										 scene='FloorPlan'+str(self.floor_plan), gridSize=self.config.gridSize[self.floor_plan],
										 snapToGrid=self.config.snapToGrid, continuous = True,
										 rotateStepDegrees=self.config.rotateStepDegrees, renderDepthImage=False,
										 renderInstanceSegmentation=self.renderInstanceSegmentation,
										 width=self.config.img_dim[2], height=self.config.img_dim[1],
										 fieldOfView=self.config.fieldOfView)

		else:
			# reset variables
			self.episodeCounter = self.episodeCounter + 1
			self.done = 0
			self.reward = 0
			self.terminated = False
			self.envStepCounter = 0
			self.episodeReward = 0.0

			self.controller.reset(scene='FloorPlan'+str(self.floor_plan), gridSize=self.config.gridSize[self.floor_plan])
			self.controller.step('ResetObjectFilter')
			self.objMeta={}
			self.visibility = {}


		if self.floor_plan not in self.reachablePositions:
			pos=self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
			self.reachablePositions[self.floor_plan]=[]
			self.robotY[self.floor_plan]=pos[0]['y']
			for item in pos:
				self.reachablePositions[self.floor_plan].append((item['x'], item['z']))

			l=np.array(self.reachablePositions[self.floor_plan])
			self.min_xz[self.floor_plan] = np.min(l, axis=0)
			max_xz = np.max(l, axis=0)
			self.get_occupancy_grid(self.config.gridSize[self.floor_plan], max_xz)
		self.setupTask()


		ret=self.gen_obs()

		if self.config.use3rdCam:
			self.update3rdCam("Add")

		return ret[0]

	def saveManualPairs(self):
		# used to save data pair when doing manual collection
		filePath = os.path.join(self.config.pretextDataDir[0], 'train')
		if not os.path.isdir(filePath):
			os.makedirs(filePath)

		datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
		filePath = os.path.join(filePath, 'data_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.pickle')

		with open(filePath, 'wb') as f:
			pickle.dump(self.saved_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
		self.saved_pairs.clear()


	def update3rdCam(self, op):
		theta=np.deg2rad(self.agentMeta['rotation']['y'])

		R_sb=np.array([[np.cos(theta), 0, np.sin(theta)],
					   [0, 1, 0],
					   [-np.sin(theta), 0, np.cos(theta)]])
		xz_trans=np.matmul(R_sb, np.expand_dims([0.,1.1,-0.8], -1))

		event = self.controller.step(
			action=op+"ThirdPartyCamera",
			position=dict(x=self.agentMeta['position']['x']+float(xz_trans[0]),
						  y=self.agentMeta['position']['y']+float(xz_trans[1]),
						  z=self.agentMeta['position']['z']+float(xz_trans[2])),
			rotation=dict(x=35, y=self.agentMeta['rotation']['y'],
						  z=self.agentMeta['rotation']['z']),
			fieldOfView=self.config.fieldOfView
		)

		if not event.metadata['lastActionSuccess']:
			warnings.warn(event.metadata['errorMessage'])


	def get_pos_act(self, obj_in_view):
		if len(self.config.allTasks[self.task.loc][obj_in_view]) == 1:
				act = self.config.allTasks[self.task.loc][obj_in_view][0]
		else:
			# check the current state of the obj_in_view and choose the opposite
			if obj_in_view=='Pillow':
				return 'PickupObject'

			if self.checkTaskDone(): #choose the same
				if self.objMeta[obj_in_view]["isToggled"]:
					act = 'ToggleObjectOn'
				else:
					act = 'ToggleObjectOff'
			else:
				if self.objMeta[obj_in_view]["isToggled"]:
					act = 'ToggleObjectOff'
				else:
					act = 'ToggleObjectOn'
		return act

	def get_negatives(self, empty, ground_truth):
		rng = self.taskLocRange[self.task.loc]
		neg_taskID = self.np_random.randint(low=rng[0], high=rng[1]) # 0,1,2,3,4
		if not empty:
			if ground_truth == neg_taskID:
				neg_taskID = self.config.taskNum

		return neg_taskID

	def get_positive_negative(self, get_negative, generate_audio):
		"""

		:param get_negative: in RSI2 pretext env, we should get negatives, which are not needed in RSI3
		:param generate_audio: audio might not be needed for pretext envs (audios could be paired in dataset.py).
		It should be True for the current sound
		:return: sound_positive, sound_negative, ground_truth, positive_audio, intent_negative
		"""
		sound_positive, sound_negative, positive_audio, intent_negative = None, None, None, None

		num_visible=0
		obj_in_view=None
		for k in self.visibility:

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
					if intent_negative==self.config.taskNum:
						sound_negative=np.zeros(shape=self.config.sound_dim)
					else:
						neg_tsk = self.taskList[intent_negative]
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

	def saveEpisodeImage(self, image):
		if self.config.episodeImgSaveInterval > 0 and self.episodeCounter % self.config.episodeImgSaveInterval == 0:
			# save the images
			imgSave = cv2.resize(image, (self.config.episodeImgSize[1], self.config.episodeImgSize[0]))
			if self.config.episodeImgSize[2] == 3:
				imgSave = cv2.cvtColor(imgSave, cv2.COLOR_RGB2BGR)
			fileName = str(self.givenSeed) + 'out' + str(self.envStepCounter) + '.jpg'
			cv2.imwrite(os.path.join(self.config.episodeImgSaveDir, fileName), imgSave)

	def gen_obs(self):
		"""
		:return: a dict containing various type of observations
		"""


		# update object metadata
		self.updateObjMeta(list(self.objMeta.keys()))
		self.checkVisible()
		self.agentMeta=self.controller.last_event.metadata["agent"]

		rgb_image=self.controller.last_event.frame
		self.saveEpisodeImage(rgb_image)

		image=rgb_image
		image=cv2.resize(rgb_image, (96, 96))

		s=[self.agentMeta['position']['x'], self.agentMeta['position']['z']]

		self.local_occupancy=self.get_local_occupancy_map(x=s[0], z=s[1], y=self.agentMeta['rotation']['y'])
		

		# sound_positive: the current sound heard by the agent
		# sound_negative: the sound that is not the current heard sound
		# current_sound_label: the ground truth label for the sound heard by the agent
		sound_positive, sound_negative, current_sound_label, positive_audio, intent_negative = \
			self.get_positive_negative(get_negative=False, generate_audio=True)


		if self.envStepCounter==0: # prepare the goal sound
			if self.config.RLTrain or self.config.render:
				# select an audio according to the task
				self.goal_sound, self.goal_audio, self.transcription=self.audio.getAudioFromTask(self.np_random, self.task, Task)
			else:
				self.goal_sound, self.goal_audio, self.transcription=self.audio.getAudioFromTask(self.np_random, self.task, Task)

			if self.config.render or self.config.RLTrain == False:
				if self.goal_audio is not None and self.config.render:
					sd.play(self.goal_audio, self.audio.fs)
				print('Goal intent is', self.task.loc+' '+self.task.obj+' '+self.task.act)
		else:
			self.goal_sound=np.ones_like(self.goal_sound)*np.inf

		if self.config.render and positive_audio is not None:
			sd.play(positive_audio, self.audio.fs)

		obs = {
			'image': np.transpose(image, (2, 0, 1)),
			'occupancy':np.transpose(np.expand_dims(self.local_occupancy, -1), (2,0,1)),
			'goal_sound': self.goal_sound,
			'current_sound': sound_positive,
			'goal_sound_label': self.taskID,
			'goal_sound_feat': np.zeros((self.config.representationDim,)),
			'image_feat': np.zeros((self.config.representationDim,)),
		}

		return obs, sound_positive, sound_negative

	def special_action(self, action_str):
		if action_str in ["ToggleObjectOn", "ToggleObjectOff"]:
			obj_in_view=None
			for k in self.visibility:
				if k!='Pillow':
					if self.visibility[k]:
						obj_in_view=k
			if obj_in_view is not None:
				e = self.controller.step(action=action_str, objectId=self.objMeta[obj_in_view]["objectId"])
		elif action_str in ["PickupObject"]:
			if 'Pillow' in self.objMeta:
				dis=self.objMeta['Pillow']['distance']
				#if dis<1.5:
				e=self.controller.step(
					action="PickupObject",
					objectId=self.objMeta['Pillow']["objectId"],
					forceAction=False,
					manualInteract=False
				)
		else:
			raise NotImplementedError

	def keyboardControl(self):

		k = self.get_term_character()
		if k in self.config.keyBoardMapping:
			action_str = self.config.keyBoardMapping[k]
			self.exeAction(action_str)

		return k


	def randomTeleport(self):
		while True:
			idx = self.np_random.randint(len(self.reachablePositions[self.floor_plan]))
			position = self.reachablePositions[self.floor_plan][idx]
			r = np.arange(0, 360, self.config.rotateStepDegrees)
			event = self.controller.step(action="Teleport",
										 position=dict(x=position[0], y=self.robotY[self.floor_plan], z=position[1]),
										 rotation=dict(x=0, y=self.np_random.choice(r), z=0), horizon=0, standing=True)
			if event.metadata['lastActionSuccess']:
				break

	def pickUpByTask(self, tsk):
		# used in pretext envs to pickup an interested object
		event=self.controller.step(
			action="PickupObject",
			objectId=self.objMeta[tsk.obj]['objectId'],
			forceAction=True,
			manualInteract=False
		)

	def exeAction(self, action_str):
		if action_str not in ["ToggleObjectOn", "ToggleObjectOff", "PickupObject"]: # if a navigational action
			self.controller.step(action=action_str)
		else:
			self.special_action(action_str)

	def step(self, action):
		act=[]
		infoDict = {}

		if self.config.RLManualControl:
			self.keyboardControl()

		else:
			action_str = self.config.allActions[int(action)]
			self.exeAction(action_str)


		self.controller.step("Pass")  # fix the design choice that images from the Unity window lag by 1 step
		# update counters
		self.envStepCounter = self.envStepCounter + 1
		# get new obs
		obs, sound_positive, sound_negative = self.gen_obs()

		if self.config.use3rdCam:
			self.update3rdCam("Update")

		r =self.rewards() # calculate reward
		self.reward = sum(r)
		self.episodeReward = self.episodeReward + self.reward
		self.done = self.termination()


		if not self.config.RLTrain:
			if self.checkTaskDone():
				self.goal_area_count = self.goal_area_count + 1
			if self.done:
				infoDict['goal_area_count']=self.goal_area_count
				print('goal area count-------------------------', self.goal_area_count)
				self.goal_area_count = 0


		return obs, self.reward, self.done, infoDict # reset will be called if done

	def checkTaskDone(self):
		if self.task.obj in ['FloorLamp', 'Television']:
			if self.task.act=='ToggleObjectOn':
				return True if self.objMeta[self.task.obj]["isToggled"] else False
			elif self.task.act=='ToggleObjectOff':
				return False if self.objMeta[self.task.obj]["isToggled"] else True
		elif self.task.obj=='Pillow':
			if self.task.act=='PickupObject':
				return True if self.objMeta[self.task.obj]['isPickedUp'] else False
		elif self.task.obj in ['Microwave', 'Fridge']:
			return True if self.objMeta[self.task.obj]['objectId'] in \
						   self.objMeta[self.task.obj]['receptacleObjectIds'] else False
		else:
			raise NotImplementedError

	def rewards(self, *args):
		rew=0.

		return [rew]

	def termination(self):
		done=False
		if self.envStepCounter >= self.maxSteps:
			done=True

		return done

	def render(self, mode='human'):
		if self.config.render:
			self.envAxShow.set_data(self.controller.last_event.frame)
			self.envAxText.set_text(self.transcription)

			self.envFig.canvas.draw_idle()
			self.envFig.canvas.start_event_loop(0.001)

			self.mapAxShow.set_data(self.local_occupancy)
			self.mapAxShow.autoscale() # if the scale is not right, it won't draw
			self.mapFig.canvas.draw_idle()
			self.mapFig.canvas.start_event_loop(0.001)

			if self.config.use3rdCam:
				self.AxShow3rd.set_data(self.controller.last_event.third_party_camera_frames[0])
				self.Fig3rd.canvas.draw_idle()
				self.Fig3rd.canvas.start_event_loop(0.001)


			time.sleep(0.1)


	def seed(self, seed=None):
		# use a random seed if seed is None
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		self.givenSeed=seed

		print("Created a AI2Thor Env with seed", seed)
		return [seed]

	def close(self):
		if self.controller:
			self.controller.stop()

	def get_term_character(self):
		fd = sys.stdin.fileno()
		old_settings = termios.tcgetattr(fd)
		try:
			tty.setraw(sys.stdin.fileno())
			ch = sys.stdin.read(1)
		finally:
			termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
		return ch


