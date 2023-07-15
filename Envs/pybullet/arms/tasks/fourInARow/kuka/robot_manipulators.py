from Envs.pybullet.arms.robot_bases import *
import numpy as np
import os
import cv2
import pybullet_data

class Kuka(BaseRobot):
	"""
	The base class for Kuka robot
	"""
	def __init__(self, config):
		model_file=os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/kuka_with_gripper2.sdf")
		super(Kuka, self).__init__(model_file=model_file, robot_name=config.robotName, scale=config.robotScale)
		self._p = None  # bullet client
		self.config=config

		self.numJoints=None
		self.desiredEndEffectorPos = [0.0, 0.0, 0.0] # the RL planner's decision will change this vector

		self.rayIDs = None


	def robot_specific_reset(self, eePositionX, eePositionY, eePositionZ):
		# reset robot
		jointPositionsReset = [0.0, 0.4, 0.0, -1.57, 0.0, 1.1, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

		if self.numJoints is None: self.numJoints=self._p.getNumJoints(self.robot_ids)

		for jointIndex in range(self.numJoints):
			self._p.resetJointState(self.robot_ids, jointIndex, jointPositionsReset[jointIndex])
			self._p.setJointMotorControl2(self.robot_ids, jointIndex, self._p.POSITION_CONTROL,
										  targetPosition=jointPositionsReset[jointIndex],
										  force=self.config.positionControlMaxForce)



		eePosition = [eePositionX, eePositionY, eePositionZ]
		orn = self._p.getQuaternionFromEuler([0, -np.pi, 0])

		jointPositionsInitial = self.invKin(eePosition, orn)
		fingerAnglePositions = [0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		jointPositionsInitial.extend(fingerAnglePositions)

		for jointIndex in range(self.numJoints):
			self._p.resetJointState(self.robot_ids, jointIndex, jointPositionsInitial[jointIndex])
			self._p.setJointMotorControl2(self.robot_ids, jointIndex, self._p.POSITION_CONTROL,
										  targetPosition=jointPositionsInitial[jointIndex],
										  force=self.config.positionControlMaxForce)

		self.desiredEndEffectorPos = [eePositionX, eePositionY, eePositionZ]


	def calc_state(self):
		eeState = self._p.getLinkState(self.robot_ids, self.config.endEffectorIndex)[0]

		s = {'eeState':eeState}
		return s

	def applyActionPretext(self, action, np_random=None):
		dx, dy, dz = 0.0, 0.0, 0.0
		ret_key_codes=None
		if self.config.pretextManualControl:
			event = p.getKeyboardEvents()
			if len(event) != 0:
				key_codes = list(event.keys())[0]
				if event[key_codes] & p.KEY_WAS_RELEASED:  # a key is pressed and released
					ret_key_codes = key_codes
					if p.B3G_UP_ARROW == key_codes:
						dx = -0.02
					if p.B3G_DOWN_ARROW == key_codes:
						dx = 0.02
					if p.B3G_LEFT_ARROW == key_codes:
						dy = -0.02
					if p.B3G_RIGHT_ARROW == key_codes:
						dy = 0.02
		elif self.config.pretextCollection:
			dx = np_random.uniform(-0.3, 0.3)
			dy = np_random.uniform(-0.4, 0.4)

		else:
			raise NotImplementedError

		self.go2desired(dx, dy, dz, controlMethod = self.config.pretextRobotControl)
		if ret_key_codes is not None:
			ret_key_codes=chr(ret_key_codes)
		return ret_key_codes

	def go2desired(self, dx, dy, dz, controlMethod):
		self.desiredEndEffectorPos[0] = self.desiredEndEffectorPos[0] + dx
		self.desiredEndEffectorPos[0] = np.clip(self.desiredEndEffectorPos[0], a_min=self.config.xMin,
												a_max=self.config.xMax)

		self.desiredEndEffectorPos[1] = self.desiredEndEffectorPos[1] + dy
		self.desiredEndEffectorPos[1] = np.clip(self.desiredEndEffectorPos[1], a_min=self.config.yMin,
												a_max=self.config.yMax)

		self.desiredEndEffectorPos[2] = self.desiredEndEffectorPos[2] + dz
		orn = self._p.getQuaternionFromEuler([0, -np.pi, 0])

		jointPositions = self.invKin(self.desiredEndEffectorPos, orn)

		if controlMethod == 'position':
			for i in range(self.config.endEffectorIndex + 1):
				self._p.setJointMotorControl2(bodyUniqueId=self.robot_ids, jointIndex=i,
											  controlMode=self._p.POSITION_CONTROL,
											  targetPosition=jointPositions[i], targetVelocity=0,
											  force=self.config.positionControlMaxForce,
											  positionGain=self.config.positionControlPositionGain,
											  velocityGain=self.config.positionControlVelGain)
			# fingers
			self._p.setJointMotorControl2(self.robot_ids, 7, self._p.POSITION_CONTROL,
										  targetPosition=0.0,
										  force=self.config.positionControlMaxForce)
			self._p.setJointMotorControl2(self.robot_ids, 8, self._p.POSITION_CONTROL, targetPosition=0,
										  force=self.config.fingerAForce)
			self._p.setJointMotorControl2(self.robot_ids, 11, self._p.POSITION_CONTROL, targetPosition=0,
										  force=self.config.fingerBForce)

			self._p.setJointMotorControl2(self.robot_ids, 10, self._p.POSITION_CONTROL, targetPosition=0,
										  force=self.config.fingerTipForce)
			self._p.setJointMotorControl2(self.robot_ids, 13, self._p.POSITION_CONTROL, targetPosition=0,
										  force=self.config.fingerTipForce)

		else:
			raise NotImplementedError

	def applyAction(self, action, np_random=None):
		dx, dy, dz = 0.0, 0.0, 0.0
		ret_key_codes=None
		if self.config.RLManualControl:
			event = p.getKeyboardEvents()
			if len(event) != 0:
				key_codes = list(event.keys())[0]
				if event[key_codes] & p.KEY_WAS_RELEASED:  # a key is pressed and released
					ret_key_codes = key_codes
					if p.B3G_UP_ARROW == key_codes:
						dx=-0.02
					if p.B3G_DOWN_ARROW == key_codes:
						dx=0.02
					if p.B3G_LEFT_ARROW == key_codes:
						dy=-0.02
					if p.B3G_RIGHT_ARROW == key_codes:
						dy=0.02

		else:
			dv = 0.02
			dx = float(np.clip(action[0],-1,+1)) * dv  # move in x direction
			dy = float(np.clip(action[1],-1,+1)) * dv  # move in y direction
			dz = 0.

		self.go2desired(dx, dy, dz, controlMethod = self.config.RLRobotControl)

		return ret_key_codes

	def get_image(self, externalCamEyePosition, externalCamTargetPosition, renderer):

		view_matrix = \
			self._p.computeViewMatrix (cameraEyePosition=externalCamEyePosition,
									cameraTargetPosition=externalCamTargetPosition,
									cameraUpVector=[0,0,1])

		proj_matrix = self._p.computeProjectionMatrixFOV(
			fov=self.config.robotFov, aspect=4. / 3.,
			nearVal=0.01, farVal=100)

		(_, _, px, _, _) = self._p.getCameraImage(
			width=self.config.robotCamRenderSize[1], height=self.config.robotCamRenderSize[0], viewMatrix=view_matrix,
			projectionMatrix=proj_matrix, shadow=0,
			# if you load the weights, do not forget to use TINY_RENDERER, otherwise the images will be different
			renderer=renderer,
			flags=self._p.ER_NO_SEGMENTATION_MASK
		)
		rgb_array = np.array(px)
		img = rgb_array[:, :, :3]
		img = img[:, 12:87, :]
		img = cv2.resize(img, (self.config.img_dim[2], self.config.img_dim[1]))

		# process the image
		if self.config.robotCamRenderSize[2] == 1:  # if we need grayscale image
			img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			img = np.reshape(img, [self.config.robotCamRenderSize[0], self.config.robotCamRenderSize[1], 1])

		return img

	def ray_test(self, objUidList):
		eePosition = list(self._p.getLinkState(self.robot_ids, self.config.endEffectorIndex)[0])
		eePosition[-1]=eePosition[-1]-0.2 # an offset so that the ray does not contact with the end effector
		rayTo=[eePosition[0], eePosition[1], self.config.tablePosition[-1]]
		results = p.rayTest(eePosition, rayTo)

		hitObjectUid = results[0][0]
		if self.config.render:
			if self.rayIDs is None:  # draw these rays out
				self.rayIDs=p.addUserDebugLine(eePosition, rayTo, self.config.rayMissColor, lineWidth=5)

			if hitObjectUid not in objUidList:
				hitPosition = [0, 0, 0]
				p.addUserDebugLine(eePosition, rayTo, self.config.rayMissColor, lineWidth=5, replaceItemUniqueId=self.rayIDs)
			else:
				hitPosition = results[0][3]
				p.addUserDebugLine(eePosition, hitPosition, self.config.rayHitColor, lineWidth=5, replaceItemUniqueId=self.rayIDs)
		return results[0][0]

	def invKin(self, pos,orn):
		# calculate inverse kinematics
		if self.config.ik_useNullSpace:
			if self.config.ik_useOrientation:
				jointPositions = self._p.calculateInverseKinematics(self.robot_ids, self.config.endEffectorIndex, pos, orn,
																lowerLimits=self.config.ik_ll, upperLimits=self.config.ik_ul,
																jointRanges=self.config.ik_jr, restPoses=self.config.ik_rp)
			else:
				jointPositions = self._p.calculateInverseKinematics(self.robot_ids, self.config.endEffectorIndex, pos,
														  lowerLimits=self.config.ik_ll, upperLimits=self.config.ik_ul,
														  jointRanges=self.config.ik_jr, restPoses=self.config.ik_rp)
		else:
			if self.config.ik_useOrientation:
				jointPositions = self._p.calculateInverseKinematics(self.robot_ids, self.config.endEffectorIndex, pos, orn,
														  jointDamping=self.config.ik_jd)
			else:
				jointPositions = self._p.calculateInverseKinematics(self.robot_ids, self.config.endEffectorIndex, pos)


		return list(jointPositions)

	def setup_scene(self, env):
		env.table_path=os.path.join(pybullet_data.getDataPath(), "table/table.urdf")












