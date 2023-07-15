import numpy as np
import torch
import torch.nn as nn
from models.ppo.utils import init
from ..ppo.model import NNBase, Flatten, get_layer_output_shape

def buildCNN(config=None):
	if config.img_dim[-1]!=96:
		modules = [
			nn.Conv2d(3, 64, 7, stride=2, padding=1), nn.ReLU(),  # (3, 120, 160)->(32, 58, 78)
			nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
			nn.MaxPool2d(2, stride=2),  # (32, 58, 78)->(32, 29, 39)
			nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(),
			nn.MaxPool2d(2, stride=2),  # (64, 29, 39)->(64, 14, 19)
			nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.ReLU(),
			nn.MaxPool2d(2, stride=2),  # (128, 14, 19))->(128, 7, 9)
			nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.ReLU(),
			nn.MaxPool2d(2, stride=2), # (256, 7, 9))->(256, 3, 4)
		]

	else:
		modules = [
            nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.ReLU(),  # (3, 96, 96)->(32, 96, 96)
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (32, 96, 96)->(32, 48, 48)
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),  # (32, 48, 48)->(64, 48, 48)
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (64, 48, 48)->(64, 24, 24)
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(),  # (64, 24, 24)->(128, 24, 24)
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (128, 24, 24))->(128, 12, 12)
            nn.Conv2d(128, 256, 3, stride=2, padding=0), nn.ReLU(),  # (128, 12, 12)->(256, 5, 5)
            nn.Conv2d(256, 128, 3, stride=1, padding=0), nn.ReLU(),  # (256, 5, 5)->(128, 3, 3)
		]

	modules.append(Flatten())

	return nn.Sequential(*modules)


class armNet_VAR(NNBase):
	def __init__(self, num_inputs, config=None, recurrent=False, recurrentInputSize=128, recurrentSize=128, actionHiddenSize=128):
		super(armNet_VAR, self).__init__(recurrent, recurrentInputSize, recurrentSize, actionHiddenSize)
		self.config = config

		init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
							   constant_(x, 0), nn.init.calculate_gain('relu'))


		self.imgCNN = buildCNN(config)
		self.imgCNN_outputShape = get_layer_output_shape(self.imgCNN, self.config.img_dim)
		init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
							   constant_(x, 0), np.sqrt(2))

		self.motorMlp = nn.Sequential(
			init_(nn.Linear(config.representationDim + config.robotStateDim, 256)), nn.ReLU(),
			init_(nn.Linear(256, 512)), nn.ReLU(),
			init_(nn.Linear(512, 256)), nn.ReLU(),
		)


		self.cnnMlp = nn.Sequential(
			init_(nn.Linear(int(np.prod(self.imgCNN_outputShape)), 512)), nn.ReLU(),
			init_(nn.Linear(512, 256)), nn.ReLU())

		self.imgMotorMlp = nn.Sequential(
			init_(nn.Linear(256, 256)), nn.ReLU(),
			init_(nn.Linear(256, recurrentInputSize)), nn.ReLU(),
		)
		self.imgMotorMlp2 = nn.Sequential(
			init_(nn.Linear(recurrentSize, 256)), nn.ReLU(),
		)
		self.soundMlp = nn.Sequential(
			init_(nn.Linear(config.representationDim, 128)), nn.ReLU(),
			init_(nn.Linear(128, 256)), nn.ReLU(),
			init_(nn.Linear(256, 256)), nn.ReLU(),
		)

		self.fusionMlp = nn.Sequential(
			init_(nn.Linear(256, 512)), nn.ReLU(),
			init_(nn.Linear(512, 256)), nn.ReLU(),

		)

		self.mlp_all = nn.Sequential(
			init_(nn.Linear(256, 256)), nn.ReLU(),
			init_(nn.Linear(256, 128)), nn.ReLU(),
		)

		self.actor = nn.Sequential(
			init_(nn.Linear(128, 128)), nn.ReLU(),
			init_(nn.Linear(128, actionHiddenSize)), nn.ReLU())

		self.critic = nn.Sequential(
			init_(nn.Linear(128, 128)), nn.ReLU(),
			init_(nn.Linear(128, 128)), nn.ReLU())

		self.critic_linear = init_(nn.Linear(128, 1))

		self.train()

	def forward(self, inputs, rnn_hxs, masks, **kwargs):
		x = inputs
		# batchSize=list(x.size())[0]
		robot_pose = x['robot_pose']
		image_feat = x['image_feat']

		motor_imgEmb = torch.cat([image_feat, robot_pose], dim=1)  # image embedding seems important for the learning
		# motor_imgEmb = robot_pose
		sound = x['goal_sound_feat']
		image = x['image']
		# image=torch.reshape(image,(batchSize, 3,96,96))
		image = self.imgCNN(image)
		image_flatten = self.cnnMlp(image)
		motor = self.motorMlp(motor_imgEmb)
		imageMotor = self.imgMotorMlp(image_flatten + motor)

		if self.is_recurrent:
			imageMotor, rnn_hxs = self._forward_gru(imageMotor, rnn_hxs, masks)

		imageMotorRnn = self.imgMotorMlp2(imageMotor)

		sound = self.soundMlp(sound)
		fusion = sound + image_flatten

		fusion = self.fusionMlp(fusion)

		final_fusion = fusion + imageMotorRnn
		x = self.mlp_all(final_fusion)

		hidden_critic = self.critic(x)
		hidden_actor = self.actor(x)

		return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs, {}



