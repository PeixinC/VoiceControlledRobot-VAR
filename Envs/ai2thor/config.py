import os
from models.pretext.ai2thor_pretext_model import VARPretextNet
from dataset import VARFineTuneDataset, VARDataset
from cfg import configBase
import sys

class AI2ThorConfig(configBase):
	def __init__(self):
		self.name = self.__class__.__name__
		self.pretext_RL=os.path.basename(sys.argv[0])
		super(AI2ThorConfig, self).__init__()

		""" Visualization settings """
		self.render = False
		self.use3rdCam = False
		self.renderUnity = True
		self.realTimeVec = False

		""" VAR setting """
		self.pretextTrain = True
		self.pretextCollection = True
		self.pretextManualControl = False
		self.pretextManualCollect = False
		self.pretextCollectNum = [100, 100, 100, 100, 100]
		self.pretextDataHasSound = False
		self.pretextModelFineTune = False

		self.pretextDataDir = ['data/pretext_training/default_finetune', ]

		self.pretextDataFileLoadNum = ['all', ]

		self.pretextDataset = VARFineTuneDataset if self.pretextModelFineTune else VARDataset
		self.pretextModel = VARPretextNet
		self.pretextModelSaveDir = os.path.join('data', 'pretext_model','default')
		self.pretextModelLoadDir = os.path.join('data', 'pretext_model','default', '39.pt')
		self.pretextModelSaveInterval = 10
		self.pretextDataNumWorkers = 8

		self.pretextDataEpisode = 200
		self.pretextDataNumFiles = 20
		self.pretextTrainBatchSize = 128
		self.pretextTestBatchSize = 128
		self.pretextLR = 1e-4
		self.pretextAdamL2 = 1e-6
		self.pretextLRStep = 'step'
		self.pretextEpoch = 40
		self.pretextLRDecayEpoch = [20, 30]
		self.pretextLRDecayGamma = 0.2
		self.representationDim = 3
		self.tripletMargin = 1.0
		self.pretextTestMethod = 'plot'
		self.plotRepresentation = 50
		self.plotNumBatch = 7
		self.annotateLastBatch = False
		self.plotRepresentationExtra = False
		self.plotExtraPath = os.path.join('data', 'episodeRecord', 'extra')
		# pretext env configuration
		self.pretextEnvName = 'ai2thor-pretext-v2'
		self.pretextEnvMaxSteps = 15
		self.pretextEnvSeed = 977
		self.pretextNumEnvs = 4 if not self.render else 1
		self.pretextVisibilityDistance = 100.

		""" RL setting """
		self.RLTrain = True
		self.RLManualControl = False
		self.RLManualControlLoaded = False
		if self.realTimeVec: self.RLManualControlLoaded=True
		self.RLModelFineTune = False
		self.RLLogDir = os.path.join('data', 'RL_model', 'ai2thor')
		self.RLPolicyBase = 'ai2thor_VAR'
		self.RLGamma = 0.99
		self.RLRecurrentPolicy = True
		self.RLLr = 6e-5
		self.RLEps = 1e-5
		self.RLMaxGradNorm = 0.5
		self.RLTotalSteps = 1e6
		self.RLModelSaveInterval = 200
		self.RLLogInterval = 100
		self.RLModelSaveDir = os.path.join('data', 'RL_model', 'default')
		self.RLModelLoadDir = os.path.join('data', 'RL_model', 'default','00000.pt')
		self.RLUseProperTimeLimits = False
		self.RLRecurrentSize = 1024
		self.RLRecurrentInputSize = 128
		self.RLActionHiddenSize = 128
		# RL env configuration
		self.RLEnvMaxSteps = 50
		self.RLRewardSoundSound = False
		self.RLEnvName = 'ai2thor-RL-v2'
		self.RLEnvSeed = 349
		self.RLNumEnvs = 8 if not self.render else 1
		self.RLVisibilityDistance = 1.5
		self.RLVisibleGrid = 9
		self.RLObsIgnore = {'current_sound', 'goal_sound',
							'goal_sound_label', }
		self.episodeImgSaveDir = os.path.join('', 'data', 'episodeRecord','tempImgs')
		self.episodeImgSaveInterval = -1
		self.episodeImgSize = (96 * 5, 96 * 5, 3)
		# ppo algorithm settings
		self.ppoClipParam = 0.2
		self.ppoEpoch = 4
		self.ppoNumMiniBatch = 2
		self.ppoValueLossCoef = 0.5
		self.ppoEntropyCoef = 0.01
		self.ppoUseGAE = True
		self.ppoGAELambda = 0.95
		self.ppoNumSteps = self.RLEnvMaxSteps
		# test RL policy
		self.success_threshold = 1
		self.RLDeterministic = True
		self.skillInfos = [{
			'path': os.path.join('data', 'RL_model', 'default', '00000.pt'), 'actionDim': 8, 'actionOffset': 0},
		]



		""" Sound command and env settings """

		self.sound_dim=(1, 600, 40)
		self.commonMediaPath = os.path.join('commonMedia')

		self.soundSource = {'dataset': 'FSC',
							'train_test': 'train',
							'FSC_max_sound_dur': 6.,
							'size': 1000,
							'FSC_obj_act': {
								'lights': ['activate', 'deactivate'],
								'music': ['activate', 'deactivate'],
								'lamp': ['activate', 'deactivate'],
							},
							'FSC_locations': ['none'],
							}
		self.soundSource['FSC_csv'] = self.soundSource['train_test'] + '_data.csv'


		self.trainingRoom=[201, 202, 203, 204, 205, 206, 207, 208, 209,
					  210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220]
		self.testingRoom=[226,227,228,229,230]

		self.allScene={'livingRoom':self.trainingRoom}



		self.cfg_check()





