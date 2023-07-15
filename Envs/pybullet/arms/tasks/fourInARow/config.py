import numpy as np
import os
from models.pretext.arm_pretext_model import VARPretextNet
from dataset import VARDataset, VARFineTuneDataset

from cfg import configBase
import sys

class ArmConfig(configBase):
	def __init__(self):
		self.name = self.__class__.__name__
		self.pretext_RL = os.path.basename(sys.argv[0])
		super(ArmConfig, self).__init__()

		""" Visualization settings """
		self.render = False  # start pybullet GUI
		self.realTimeVec = False

		""" VAR setting """
		self.pretextTrain = True
		self.pretextCollection = True
		self.pretextManualCollect = False
		self.pretextManualControl = False
		self.pretextDataDir = ['data/pretext_training/default',]
		self.pretextCollectNum = [50, 50, 50, 50, 100]
		self.pretextDataHasSound = False
		self.pretextModelFineTune = True
		self.pretextDataset = VARFineTuneDataset if self.pretextModelFineTune else VARDataset
		self.pretextDataFileLoadNum = ['all', 'all', 'all']
		self.pretextModel = VARPretextNet
		self.pretextModelSaveDir = os.path.join('data', 'pretext_model', 'default')
		self.pretextModelLoadDir = os.path.join(self.pretextModelSaveDir,'39.pt')
		self.pretextModelSaveInterval = 10
		self.pretextDataNumWorkers = 4
		self.pretextDataEpisode = 500
		self.pretextDataNumFiles = 20
		self.pretextTrainBatchSize = 128
		self.pretextTestBatchSize = 128
		self.pretextLR = 1e-4
		self.pretextAdamL2 = 1e-6
		self.pretextLRStep = 'step'
		self.pretextEpoch = 40
		self.pretextLRDecayEpoch = [10, 30, 50]
		self.pretextLRDecayGamma = 0.2
		self.representationDim = 3
		self.tripletMargin = 1.0
		self.plotRepresentation = 50
		self.plotNumBatch = 10
		self.annotateLastBatch = False
		self.plotRepresentationExtra = False  # draw datapoint for images in episodeImgSaveDir or sound in commonMedia
		self.plotExtraPath = os.path.join('data', 'episodeRecord', 'extra')
		# pretext env configuration
		self.pretextEnvName = 'arms-pretext-v2'
		self.pretextEnvMaxSteps = 30
		self.pretextEnvSeed = 453
		self.pretextNumEnvs = 4 if not self.render else 1


		""" RL setting """
		self.RLManualControl = False
		self.RLManualControlLoaded = False
		if self.realTimeVec: self.RLManualControlLoaded = True
		self.RLTrain = False
		self.RLModelFineTune = True
		self.RLPolicyBase = 'arm_VAR'
		self.RLGamma = 0.99
		self.RLRecurrentPolicy = True
		self.RLLr = 3e-5
		self.RLEps = 1e-5
		self.RLMaxGradNorm = 0.5
		self.RLTotalSteps = 3e6
		self.RLModelSaveInterval = 200
		self.RLLogInterval = 100
		self.RLObsIgnore = {'current_sound', 'goal_sound',
							'goal_sound_label', }  # the observation name that will be ignored for RL training
		self.RLModelSaveDir = os.path.join('data', 'RL_model', 'default')
		self.RLModelLoadDir = os.path.join('data', 'RL_model', 'default', '00000.pt')
		self.RLUseProperTimeLimits = False
		self.RLRecurrentSize = 512
		self.RLRecurrentInputSize = 128
		self.RLActionHiddenSize = 128
		# RL env configuration
		self.RLEnvMaxSteps = 100  # the max number of actions (decisions) for an episode. Time horizon N.
		self.RLEnvName = 'arms-RL-v2'
		self.RLEnvSeed = 40
		self.RLNumEnvs = 8 if not self.render else 1
		self.RLRewardSoundSound = False
		self.RLUseEnvReward = False
		self.episodeImgSaveDir = os.path.join('data', 'episodeRecord','tempImgs')
		self.episodeImgSaveInterval = -1
		self.episodeImgSize = (224, 224, 3)
		# ppo algorithm settings
		self.ppoClipParam = 0.2
		self.ppoEpoch = 4
		self.ppoNumMiniBatch = 2 if not self.render else 1
		self.ppoValueLossCoef = 0.5
		self.ppoEntropyCoef = 0.01
		self.ppoUseGAE = True
		self.ppoGAELambda = 0.95
		self.ppoNumSteps = self.RLEnvMaxSteps
		# test RL policy
		self.success_threshold = 1
		self.RLDeterministic = True
		self.skillInfos = [
			{'path': os.path.join('data', 'RL_model', 'default', '00000.pt'),
			 'actionDim': 2, },
		]


		""" Sound command and env settings """
		self.robotType='kuka'
		self.objSet=0 # objSet=0: ['key', 'key', 'key', 'key']
		self.commandType = 'order'
		self.commonMediaPath = os.path.join('commonMedia')

		self.soundSourcePreset='normal'
		if self.soundSourcePreset=='mix':
			self.sound_dim = (1, 100, 40)  # sound matrix dimension (1, frames, numFeat)
			self.soundSource = {'dataset': ['GoogleCommand', 'UrbanSound'],
								'items': {'GoogleCommand':['house', 'tree', 'bird', 'dog'],
										  'UrbanSound':['jackhammer', None, None, 'dog_bark'] },
								'size': {'GoogleCommand': [25, 50, 50, 25], 'UrbanSound':[25, 0, 0, 25]},
								'train_test': 'test',
								}
		elif self.soundSourcePreset=='normal':
			self.sound_dim = (1, 100, 40)  # sound matrix dimension (1, frames, numFeat)
			self.soundSource = {'dataset': ['GoogleCommand'],
								'max_sound_dur': {'GoogleCommand': 6.},
								'items': {'GoogleCommand': ['zero', 'one', 'two', 'three']},
								'size': {'GoogleCommand': [1000]*4},
								'train_test': 'train',
								}
		self.taskNum = len(self.soundSource['items'][self.soundSource['dataset'][0]])
		self.ifReset = True  # if you want to reset the scene after an episode ends




		self.cfg_check()