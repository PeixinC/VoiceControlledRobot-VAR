import torch
import torch.nn as nn
from .pretext_base import PretextNetBase

def buildSoundBranch(nn_module, config=None):
		nn_module.rnn = torch.nn.GRU(input_size=64 * 7, hidden_size=512, batch_first=True, bidirectional=True)
		nn_module.cnn = nn.Sequential(
			nn.Conv2d(1, 64, (11, 11), stride=(2, 2), padding=(5, 5)), nn.ReLU(),  # (1, 600, 40)->(32, 300, 20)
			nn.Conv2d(64, 64, (11, 5), stride=(2, 2), padding=(5, 5)), nn.ReLU(),  # (32, 300, 20)->(32, 150, 13)
			nn.Conv2d(64, 64, (7, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(),  # (32, 150, 13)->(32, 73, 7)
		)


def buildCNN(nn_module, config=None):
	modules = [
		nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.ReLU(),  # (3, 96, 96)->(32, 96, 96)
		nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ReLU(),
		nn.MaxPool2d(2, stride=2),  # (32, 96, 96)->(32, 48, 48)
		nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),  # (32, 48, 48)->(64, 48, 48)
		nn.MaxPool2d(2, stride=2),  # (64, 48, 48)->(64, 24, 24)
		nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),  # (64, 24, 24)->(64, 24, 24)
		nn.MaxPool2d(2, stride=2),  # (64, 24, 24))->(64, 12, 12)
		nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(),  # (64, 12, 12)->(128, 12, 12)
		nn.MaxPool2d(2, stride=2),  # (128, 12, 12)->(128, 6, 6)
		nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),  # (128, 6, 6)->(128, 3, 3)
		nn.Flatten()
	]


	nn_module.imgBranch = nn.Sequential(*modules)


def soundBranch(nn_module, sound):
	cnn_out = nn_module.cnn(sound)
	cnn_out = torch.reshape(torch.transpose(cnn_out, dim0=1, dim1=2), (-1, 73, 64 * 7))
	_, rnn_out = nn_module.rnn(cnn_out)
	rnn_out = torch.cat((rnn_out[0, :, :], rnn_out[1, :, :]), dim=1)
	return rnn_out


class VARPretextNet(PretextNetBase):
	def __init__(self, config):
		super(VARPretextNet, self).__init__()
		self.config=config
		self.soundBranch = soundBranch
		self.zero_feat = torch.zeros((config.representationDim,)).cuda()
		buildCNN(self, config)

		buildSoundBranch(self, config)
		self.imgTriplet = nn.Sequential(
			nn.Linear(128 * 9, 128), nn.ReLU(),
			nn.Linear(128, config.representationDim)
		)

		self.soundTriplet = nn.Sequential(nn.Linear(2 * 512, 128), nn.ReLU(),
										  nn.Linear(128, 64), nn.ReLU(),
										  nn.Linear(64, config.representationDim)
										  )


		self.cached_sound=None # the goal sound can be cached and be encoded only once

	def forward(self, image, sound_positive, sound_negative, is_train=False):
		return self.VAR_forward(image, sound_positive, sound_negative, is_train)

