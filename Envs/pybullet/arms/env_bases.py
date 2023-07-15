from Envs.pybullet.arms.scene_abstract import SingleRobotEmptyScene
import numpy as np
import pybullet as p
import gym, gym.spaces, gym.utils, gym.utils.seeding
from pybullet_utils import bullet_client

import pkgutil


class BaseEnv(gym.Env):
	"""
	Base class for Bullet physics simulation environments in a Scene.
	These environments create single-player scenes and behave like normal Gym environments, if
	you don't use multiplayer.
	"""


	def __init__(self, config, render, action_space, observation_space):
		# Pybullet related
		self.scene = None
		self.physicsClientId = -1 # at the first run, we do not own physics client
		self.ownsPhysicsClient = False
		self._p=None
		self.renderer=None
		self.config=config
		self.timeStep = 1. / 240.

		# setup GUI camera
		self.debugCam_dist=config.debugCam_dist
		self.debugCam_yaw=config.debugCam_yaw
		self.debugCam_pitch=config.debugCam_pitch

		# robot related
		self.isRender = render
		self.action_space = action_space
		self.observation_space = observation_space


		#episode related
		self.episodeCounter=-1
		self.envStepCounter=0
		self.done = 0
		self.reward = 0
		self.episodeReward = 0.0
		self.terminated = False

		self.np_random=None
		self.givenSeed=None

		# Debug
		self.logID = None

	def create_single_player_scene(self, bullet_client):
		"""
		Setup physics engine and simulation
		:param bullet_client:
		:return:
		"""

		return SingleRobotEmptyScene(bullet_client, gravity=(0, 0, -9.8),
									 timestep=self.timeStep, frame_skip=self.config.frameSkip,
									 render=self.config.render)

	def seed(self, seed=None):
		# use a random seed if seed is None
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		self.givenSeed=seed
		robotType=self.config.robotType
		print("Created a "+ robotType + "Env with seed", seed)
		return [seed]

	def reset(self):

		# starts pybullet client and create_single_player_scene
		# if it is the first run, setup Pybullet client and set GUI camera
		if self.physicsClientId < 0:
			self.ownsPhysicsClient = True

			if self.isRender:
				self._p = bullet_client.BulletClient(connection_mode=p.GUI)
			else:
				self._p = bullet_client.BulletClient()

			self._p.resetSimulation()
			self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

			# optionally enable EGL for faster headless rendering
			try:
				raise ValueError # EGL plugin is not very well supported
				con_mode = self._p.getConnectionInfo()['connectionMethod']
				if con_mode == self._p.DIRECT:
					egl = pkgutil.get_loader('eglRenderer')
					if egl:
						self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
					else:
						self._p.loadPlugin("eglRendererPlugin")

				self.renderer=p.ER_BULLET_HARDWARE_OPENGL
				print("Hardware OpenGL acceleration")
			except:
				self.renderer=p.ER_TINY_RENDERER
				print("Failed at loading EGL")

			self.physicsClientId = self._p._client
			self._p.configureDebugVisualizer(p.COV_ENABLE_GUI,0, lightPosition=[-5, -40, 200])


			self._p.resetDebugVisualizerCamera(cameraDistance=self.debugCam_dist, cameraYaw=self.debugCam_yaw,
												cameraPitch=self.debugCam_pitch, cameraTargetPosition=[0, 0, 0])
			assert self._p.isNumpyEnabled() == 1


		# if it is the first run, build scene and setup simulation physics
		if self.scene is None:
			self.scene = self.create_single_player_scene(self._p)

		# if it is not the first run
		if self.ownsPhysicsClient:
			self.scene.episode_restart()

		# reset counters
		self.episodeCounter = self.episodeCounter + 1
		self.done = False
		self.reward = 0
		self.terminated = False
		self.envStepCounter = 0
		self.episodeReward = 0.0

		obs=self.envReset()



		return obs

	def drawRectangleDebug(self, debugLines, xMin, xMax, yMin, yMax, z):
		#TODO: draw rectangle on other planes
		start = [xMin, yMin, z]
		end = [xMax, yMin, z]
		debugLines.append(p.addUserDebugLine(start, end, (0,1,0), lineWidth=5))
		start = [xMin, yMax, z]
		end = [xMax, yMax, z]
		debugLines.append(p.addUserDebugLine(start, end, (0,1,0), lineWidth=5))
		start = [xMax, yMin, z]
		end = [xMax, yMax, z]
		debugLines.append(p.addUserDebugLine(start, end, (0,1,0), lineWidth=5))
		start = [xMin, yMin, z]
		end = [xMin, yMax, z]
		debugLines.append(p.addUserDebugLine(start, end, (0,1,0), lineWidth=5))

	def drawGrid(self, gridLines, gridSize, xMin, xMax, yMin, yMax, z):
		"""
		draw a 2D grid within xMin, xMax, yMin, yMax
		"""
		x=np.arange(xMin, xMax, gridSize)
		y=np.arange(yMin, yMax, gridSize)

		for i in range(1, len(x)):
			gridLines.append(p.addUserDebugLine([x[i], yMin, z], [x[i], y[-1], z], (0,0,1), lineWidth=3))

		for i in range(1, len(y)):
			gridLines.append(p.addUserDebugLine([xMin, y[i], z], [x[-1], y[i], z], (0,0,1), lineWidth=3))


	def drawAABB(self, aabb):
		aabbMin = aabb[0]
		aabbMax = aabb[1]
		f = [aabbMin[0], aabbMin[1], aabbMin[2]]
		t = [aabbMax[0], aabbMin[1], aabbMin[2]]
		p.addUserDebugLine(f, t, [1, 0, 0])
		f = [aabbMin[0], aabbMin[1], aabbMin[2]]
		t = [aabbMin[0], aabbMax[1], aabbMin[2]]
		p.addUserDebugLine(f, t, [0, 1, 0])
		f = [aabbMin[0], aabbMin[1], aabbMin[2]]
		t = [aabbMin[0], aabbMin[1], aabbMax[2]]
		p.addUserDebugLine(f, t, [0, 0, 1])

		f = [aabbMin[0], aabbMin[1], aabbMax[2]]
		t = [aabbMin[0], aabbMax[1], aabbMax[2]]
		p.addUserDebugLine(f, t, [1, 1, 1])

		f = [aabbMin[0], aabbMin[1], aabbMax[2]]
		t = [aabbMax[0], aabbMin[1], aabbMax[2]]
		p.addUserDebugLine(f, t, [1, 1, 1])

		f = [aabbMax[0], aabbMin[1], aabbMin[2]]
		t = [aabbMax[0], aabbMin[1], aabbMax[2]]
		p.addUserDebugLine(f, t, [1, 1, 1])

		f = [aabbMax[0], aabbMin[1], aabbMin[2]]
		t = [aabbMax[0], aabbMax[1], aabbMin[2]]
		p.addUserDebugLine(f, t, [1, 1, 1])

		f = [aabbMax[0], aabbMax[1], aabbMin[2]]
		t = [aabbMin[0], aabbMax[1], aabbMin[2]]
		p.addUserDebugLine(f, t, [1, 1, 1])

		f = [aabbMin[0], aabbMax[1], aabbMin[2]]
		t = [aabbMin[0], aabbMax[1], aabbMax[2]]
		p.addUserDebugLine(f, t, [1, 1, 1])

		f = [aabbMax[0], aabbMax[1], aabbMax[2]]
		t = [aabbMin[0], aabbMax[1], aabbMax[2]]
		p.addUserDebugLine(f, t, [1.0, 0.5, 0.5])
		f = [aabbMax[0], aabbMax[1], aabbMax[2]]
		t = [aabbMax[0], aabbMin[1], aabbMax[2]]
		p.addUserDebugLine(f, t, [1, 1, 1])
		f = [aabbMax[0], aabbMax[1], aabbMax[2]]
		t = [aabbMax[0], aabbMax[1], aabbMin[2]]
		p.addUserDebugLine(f, t, [1, 1, 1])

	def envReset(self):
		raise NotImplementedError



	def render(self, mode='human'):
		# no need to implement this function
		pass


	def close(self):
		if self.ownsPhysicsClient:
			if self.physicsClientId >= 0:
				self._p.disconnect()
		self.physicsClientId = -1


	def step(self, action):
		raise NotImplementedError