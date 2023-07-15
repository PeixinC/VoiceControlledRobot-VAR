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
		self.render = False # turn on a matplotlib window to show what the robot sees.
		self.use3rdCam = False  # True: show a 3rd camera view in a window when render is True. It is for debugging
		self.renderUnity = True  # Show unity window or not. If False, run ai2thor in headless mode
		self.realTimeVec = False  # True: show real-time embedding given a VAR

		""" VAR setting """
		self.pretextTrain = True # True: train the VAR; False: test the VAR
		self.pretextCollection = True # True: collect data automatically; False: data is assumed to be on disk
		self.pretextManualControl = False # True: control the agent with keyboard
		self.pretextManualCollect = False # True: collect data manually, pretextCollection and pretextManualControl should be True as well
		self.pretextCollectNum = [100, 100, 100, 100, 100] # number of pairs to be collected for each intent
		self.pretextDataHasSound = False # True: a data pair will contain raw sound data in addition to the intent ID
		self.pretextModelFineTune = False  # True: load a trained VAR and fine-tune, self.pretextTrain must be True as well
		# REMEMBER to change pretextDataDir and pretextModelSaveDir for the fine-tuning to prevent overwriting your
		# original data and checkpoints

		self.pretextDataDir = ['data/pretext_training/default_finetune', ] # collected data will be saved to the first path in this list
		# use this list to load the collected data for the VAR training or testing

		self.pretextDataFileLoadNum = ['all', ] # 'all': load all the data in the path in self.pretextDataDir

		self.pretextDataset = VARFineTuneDataset if self.pretextModelFineTune else VARDataset
		self.pretextModel = VARPretextNet # VAR model
		self.pretextModelSaveDir = os.path.join('data', 'pretext_model','default') # path to save the VAR
		self.pretextModelLoadDir = os.path.join('data', 'pretext_model','default', '39.pt') # path to a VAR checkpoint
		self.pretextModelSaveInterval = 10 # save the checkpoint this interval
		self.pretextDataNumWorkers = 8 # pytorch data loader worker
		# the total number of episodes=pretextNumEnvs*pretextDataEpisode*pretextDataNumFiles
		self.pretextDataEpisode = 200  # each env will collect this number of episode
		self.pretextDataNumFiles = 20  # this number of pickle files will be generated
		self.pretextTrainBatchSize = 128 # batch size for the training
		self.pretextTestBatchSize = 128 # batch size for the testing
		self.pretextLR = 1e-4  # learning rate
		self.pretextAdamL2 = 1e-6
		self.pretextLRStep = 'step'  # choose from ["cos", "step", "none"]
		self.pretextEpoch = 40 # training epoch
		self.pretextLRDecayEpoch = [20, 30]  # milestones for learning rate decay
		self.pretextLRDecayGamma = 0.2  # multiplicative factor of learning rate decay
		self.representationDim = 3 # dimension of the representation
		self.tripletMargin = 1.0  # triplet loss margin
		self.pretextTestMethod = 'plot' # only plot for now
		self.plotRepresentation = 50  # plot the representation space every this number of epoch, -1 for not plotting
		self.plotNumBatch = 7 # plot this number of batch data only
		self.annotateLastBatch = False # output last batch of images to self.episodeImgSaveDir and annotate them on the plot
		self.plotRepresentationExtra = False  # draw images in self.plotExtraPath on the plot
		self.plotExtraPath = os.path.join('data', 'episodeRecord', 'extra')
		# pretext env configuration
		self.pretextEnvName = 'ai2thor-pretext-v2'
		self.pretextEnvMaxSteps = 15  # the length of an episode to collect data
		self.pretextEnvSeed = 977 # random seed
		self.pretextNumEnvs = 4 if not self.render else 1  # number of envs to collect data in parallel
		self.pretextVisibilityDistance = 100. # ai2thor visibility

		""" RL setting """
		self.RLTrain = True  # True: train the policy; False: test the policy
		self.RLManualControl = False  # True: control the agent with keyboard. self.render should be True
		self.RLManualControlLoaded = False # True: load the representation from pretextModelLoadDir during manual control
		if self.realTimeVec: self.RLManualControlLoaded=True
		self.RLModelFineTune = False  # True: load the trained policy and fine-tune the RL policy, self.RLTrain must be True
		# REMEMBER to change RLModelSaveDir for the fine-tuning to prevent overwriting your original and checkpoints
		self.RLLogDir = os.path.join('data', 'RL_model', 'ai2thor')  # useless
		self.RLPolicyBase = 'ai2thor_VAR'
		self.RLGamma = 0.99
		self.RLRecurrentPolicy = True
		self.RLLr = 6e-5  # RL learning rate
		self.RLEps = 1e-5
		self.RLMaxGradNorm = 0.5
		self.RLTotalSteps = 1e6  # total RL training steps
		self.RLModelSaveInterval = 200  # save a checkpoint every this number of RL updates
		self.RLLogInterval = 100  # print out losses every this number of RL updates
		self.RLModelSaveDir = os.path.join('data', 'RL_model', 'default')  # save the model to this path
		self.RLModelLoadDir = os.path.join('data', 'RL_model', 'default',
										   '00000.pt')  # load the model from this path for fine-tuning
		self.RLUseProperTimeLimits = False
		self.RLRecurrentSize = 1024  # the GRU hidden size in the RL policy network
		self.RLRecurrentInputSize = 128  # input size of the GRU
		self.RLActionHiddenSize = 128  # the hidden size before action output
		# RL env configuration
		self.RLEnvMaxSteps = 50  # the max number of actions (decisions) for an episode, time horizon.
		self.RLRewardSoundSound = False  # use the dot product between goal sound and current sound as reward
		self.RLEnvName = 'ai2thor-RL-v2'
		self.RLEnvSeed = 349  # RL env random seed
		self.RLNumEnvs = 8 if not self.render else 1  # number of envs to collect RL data in parallel
		self.RLVisibilityDistance = 1.5  # ai2thor visibility
		self.RLVisibleGrid = 9  # should be an odd number
		self.RLObsIgnore = {'current_sound', 'goal_sound',
							'goal_sound_label', }  # the observation name that will be ignored for RL policy update
		self.episodeImgSaveDir = os.path.join('', 'data', 'episodeRecord',
											  'tempImgs')  # output camera images to this location
		self.episodeImgSaveInterval = -1  # -1 for not saving. Save an episode of camera images every imgSaveInterval episode
		self.episodeImgSize = (96 * 5, 96 * 5, 3)  # (height, width, channel)
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
		self.RLDeterministic = True  # True: deterministic policy; False: stochastic policy
		self.skillInfos = [{
			'path': os.path.join('data', 'RL_model', 'default', '00000.pt'), 'actionDim': 8, 'actionOffset': 0},
		]  # set the path to an RL policy checkpoint for testing



		""" Sound command and env settings """

		self.sound_dim=(1, 600, 40)  # sound matrix dimension (1, frames, numFeat)
		self.commonMediaPath = os.path.join('commonMedia') # the commonMedia folder to load sound data
		# fluent speech dataset metadata
		self.soundSource = {'dataset': 'FSC',
							'train_test': 'train',
							'FSC_max_sound_dur': 6., # max duration audio clip
							'size': 1000, # load at most this number of audio clips per class to the memory
							# 1000 for training, and 50 for testing and fine-tuning
							'FSC_obj_act': {
								'lights': ['activate', 'deactivate'],
								'music': ['activate', 'deactivate'],
								'lamp': ['activate', 'deactivate'],
							},
							'FSC_locations': ['none'],
							}
		self.soundSource['FSC_csv'] = self.soundSource['train_test'] + '_data.csv' # get test_data.csv or train_data.csv

		# ai2thor 20 training living rooms: 201~220; 5 testing living rooms: 226~230
		self.trainingRoom=[201, 202, 203, 204, 205, 206, 207, 208, 209,
					  210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220]
		self.testingRoom=[226,227,228,229,230]

		self.allScene={'livingRoom':self.trainingRoom}



		self.cfg_check() # performing some configuration checks





