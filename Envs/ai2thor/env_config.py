from collections import OrderedDict


class EnvConfig(object):
	def __init__(self, x):
		"""
		The environment configuration for AI2Thor environment.
		"""
		x.envFolder = 'ai2thor'
		x.img_dim = (3, 96, 96)  # (channel, image_height, image_width)
		x.keyBoardMapping = OrderedDict(
			[
				('w', "MoveAhead"), ('s', 'MoveBack'), ('a', 'MoveLeft'), ('d', 'MoveRight'),
				('q', "RotateLeft"), ('e', "RotateRight"),
				('T', "ToggleObjectOn"), ('t', "ToggleObjectOff"),
			])
		x.allActions = list(x.keyBoardMapping.values())
		x.allTasks = OrderedDict([
			('livingRoom', OrderedDict(
				[
					('FloorLamp', ['ToggleObjectOn', 'ToggleObjectOff']),
					('Television', ['ToggleObjectOn', 'ToggleObjectOff']),
				]
			)),
		])
		x.RLActionDim = (len(x.allActions),)

		x.taskNum = 0
		for loc in x.allTasks:
			for obj in x.allTasks[loc]:
				x.taskNum = x.taskNum + len(x.allTasks[loc][obj])

		# the step size for large rooms:0.5, the step size for the small rooms: 0.25
		# the larger step size makes sure that the robot can finish a task within RLEnvMaxSteps
		x.gridSize = {201: 0.25, 202: 0.25, 203: 0.25, 204: 0.25, 205: 0.25, 206: 0.25,
						 207: 0.25, 208: 0.25, 209: 0.25, 210: 0.25, 211: 0.25, 212: 0.25,
						 213: 0.25, 214: 0.25, 215: 0.25, 216: 0.25, 217: 0.25, 218: 0.25,
						 219: 0.25, 220: 0.25, 226: 0.25, 227: 0.25, 228: 0.25, 229: 0.25, 230: 0.5}
		x.snapToGrid = False
		x.rotateStepDegrees = 45
		x.fieldOfView = 90

		# a dict that defines the relations between ai2thor and fluent speech dataset
		# key: ai2thor, val: fluent speech dataset
		x.synonym = {
			# location
			'livingRoom': ['none'],
			# object: when an object itself contains action, we write as object_action
			'FloorLamp': ['lights', 'lamp'], 'Television': ['music'],
			# action
			'ToggleObjectOn': ['increase', 'activate'], 'ToggleObjectOff': ['decrease', 'deactivate'],
		}

		# env control
		x.domainRandomization = ['randomInitialPose', 'randomObjState']