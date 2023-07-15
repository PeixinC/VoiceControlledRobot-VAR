import torch
import torch.nn as nn
from utils import Flatten
from ..ppo.model import get_layer_output_shape
import numpy as np
from .pretext_base import PretextNetBase


def buildCNN(nn_module, config=None):
	modules = [
		nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),  # (3, 96, 96)->(32, 48, 48)
		nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),  # (32, 48, 48)->(32, 24, 24)
		nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # (32, 24, 24)->(64,12,12)
		nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),  # (64, 12, 12)->(64, 6, 6)
		nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),  # (64, 12, 12)->(64, 3, 3)
		Flatten()
	]
	nn_module.imgBranch=nn.Sequential(*modules)


def buildSoundBranch(nn_module, config=None):

	modules = [nn.Conv2d(1, 32, (5, 40), stride=(2, 1)), nn.ReLU(),  # (1, 100, 40)->(32, 48, 1)
			   nn.Conv2d(32, 32, (3, 1), stride=(2, 1)), nn.ReLU(),  # (32, 48, 1)->(32, 23, 1)
			   nn.Conv2d(32, 32, (3, 1), stride=(2, 1)), nn.ReLU(),  # (32, 23, 1)->(32, 11, 1)
			   nn.Conv2d(32, 32, (3, 1), stride=(2, 1)), nn.ReLU(),  # (32, 11, 1)->(32, 5, 1)
			   Flatten()
			   ]

	nn_module.soundCNN= nn.Sequential(*modules)


def soundBranch(nn_module, sound):
	return nn_module.soundCNN(sound)


class VARPretextNet(PretextNetBase):
	def __init__(self, config):
		super(VARPretextNet, self).__init__()
		self.config=config
		self.soundBranch=soundBranch
		self.cached_sound = None  # the goal sound can be cached and be encoded only once
		buildCNN(self, config)
		buildSoundBranch(self, config)
		self.imgCNN_outputShape=get_layer_output_shape(self.imgBranch, self.config.img_dim)
		self.imgTriplet=nn.Sequential(
			nn.Linear(int(np.prod(self.imgCNN_outputShape)), 128), nn.ReLU(),
			nn.Linear(128, config.representationDim)
		)

		self.soundBranch_outputShape = self.soundBranch(self, torch.rand(*self.config.sound_dim)).data.shape
		audio_cnn_dim = int(np.prod(self.soundBranch_outputShape))
		self.soundTriplet= nn.Sequential(
			nn.Linear(audio_cnn_dim, 128), nn.ReLU(),
			nn.Linear(128, config.representationDim)
		)

	def forward(self, image, sound_positive, sound_negative, is_train=False):
		return self.VAR_forward(image, sound_positive, sound_negative, is_train=False)

