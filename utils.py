import torch.nn as nn
import os
import glob
from torch.optim.lr_scheduler import MultiStepLR
import pickle
import functools


class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

def rgetattr(oj, atr, *args):
	def _getattr(oj, atr):
		return getattr(oj, atr, *args)
	return functools.reduce(_getattr, [oj] + atr.split('.'))


def confirm_from_user(prompt):
	x = input(prompt+'[yes(y)/Any other]')
	if x == 'yes' or x == 'y': return True
	else: return False

def drawArrows(ax, fig, v_img, v_sound, quiver_img=None, quiver_sound=None):

	if quiver_img is not None:
		quiver_img.remove()
	if v_img is not None:
		v_img = v_img[0]
		quiver_img = ax.quiver(0., 0., 0., v_img[0], v_img[1], v_img[2], color='m', alpha=.6, lw=3)

	if quiver_sound is not None:
		quiver_sound.remove()
	if v_sound is not None:
		v_sound = v_sound[0]
		quiver_sound = ax.quiver(0., 0., 0., v_sound[0], v_sound[1], v_sound[2], color='m', alpha=1., lw=3)

	fig.canvas.draw_idle()
	fig.canvas.start_event_loop(0.001)
	return quiver_img, quiver_sound

def get_scheduler(config, optimizer):
	if config.pretextLRStep == "step":
		return MultiStepLR(optimizer, milestones=config.pretextLRDecayEpoch, gamma=config.pretextLRDecayGamma)
	else:
		return None


def convert_pickle_protocol(path):
	for filePath in glob.glob(os.path.join(path, '*.pickle')):
		with open(filePath, 'rb') as f:
			x = pickle.load(f)
		with open(filePath, 'wb') as f:
			pickle.dump(x, f, protocol=2)



