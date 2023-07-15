import os
import numpy as np

class EnvConfig(object):
	def __init__(self, x):

		# common env configuration
		x.objList = ['key', 'key', 'key', 'key']

		x.taskNum = len(x.objList)
		# hide the obj so that the camera can't see
		# mode can be 'random', 'fix', 'none'. If 'random', it will hide 'hideNum' blocks and 'hideIdx' is irrelevant
		# If 'fix', it will hide the blocks indicated by the list 'hideIdx' , 'hideNum' is irrelevant
		# if 'none', hide no blocks
		x.hideObj = {'mode': 'none', 'hideNum': 1, 'hideIdx': [2]}
		x.objInterval = 0.1  # the distance between two objects
		x.objXRand = [0.05, -0.05]
		x.objYRand = [0.05, -0.45]
		x.objsXRand = [0,0]  # the relative difference among objects in x direction
		x.objsYRand = [0,0]  # the relative difference among objects in y direction
		x.objZ = {'key':-0.085}
		x.tablePosition = [0.5, 0.0, -0.75]
		# object and end-effector location range
		x.xMax = 0.75
		x.xMin = 0.45
		x.yMax = 0.35
		x.yMin = -0.25
		x.img_dim = (3, 96, 96)  # (channel, image_height, image_width)


		x.frameSkip = 16
		x.rayHitColor = [1, 0, 0]
		x.rayMissColor = [0, 1, 0]

		# robot configuration
		x.robotName = 'base_link'
		x.robotStateDim=2
		x.continuousControl=True
		x.robotPosition = [-0.1, 0.0, 0.07]
		x.eeXInitRand = [0.05, -0.05]  # randomize ee position at the beginning of each episode
		x.eeYInitRand = [0.05, -0.05]
		x.robotScale = 1
		x.endEffectorHeight = 0.22
		x.RLRobotControl = 'position'
		x.pretextRobotControl = 'position'

		x.selfCollision = True
		x.endEffectorIndex = 6  # we mainly control this joint for position
		x.positionControlMaxForce = 500
		x.positionControlPositionGain = 0.03  # 0.03
		x.positionControlVelGain = 1.0  # 1.0

		x.fingerAForce = 2
		x.fingerBForce = 2
		x.fingerTipForce = 2

		# inverse kinematics settings
		x.ik_useNullSpace = True
		x.ik_useOrientation = True
		x.ik_ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]  # lower limits for null space
		x.ik_ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]  # upper limits for null space
		x.ik_jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]  # joint ranges for null space
		x.ik_rp = [0, 0, 0, 0.5 * np.pi, 0, -np.pi * 0.5 * 0.66, 0]  # restposes for null space
		x.ik_jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # joint damping coefficents

		# robot camera
		x.robotCamOffset = 0  # it is used to adjust the near clipping plane of the camera
		x.robotCamRenderSize = (75, 100, 3)  # simulation render (height, width, channel)
		x.robotFov = 48.8

		x.externalCamEyePosition = [1.2, 0, 0.3]
		x.externalCamTargetPosition = [0.6,0,0]

		# pybullet debug GUI viewing angle and distance
		x.debugCam_dist = 1.0
		x.debugCam_yaw = 90
		x.debugCam_pitch = -30


		# env information and property
		x.mediaPath = os.path.join("Envs", "pybullet", "arms", "media")  # objects' model
		x.envFolder = os.path.join('pybullet', 'arms')



		x.RLActionDim = (2,)
		x.pretextActionDim = (2,)
