import os
from python_speech_features import mfcc
from scipy.io import wavfile
import glob
import numpy as np
from collections import namedtuple
import pandas as pd
from torchaudio.transforms import MFCC as torch_mfcc
import torch


class audioLoader(object):
	def __init__(self, config):
		"""
		config: configuration file
		mem: load the audio data to memory
		"""

		self.config=config

		self.soundSource = self.config.soundSource # a dictionary containing what to load
		self.param_func=namedtuple('sound_param', ['nFFT', 'windowLenTime','windowStepTime'])
		self.param_dict={
		'GoogleCommand': self.param_func(nFFT=512, windowLenTime=0.025, windowStepTime=0.01),
		'NSynth': self.param_func(nFFT=1024, windowLenTime=0.05, windowStepTime=0.04), 
		'UrbanSound': self.param_func(nFFT=1024, windowLenTime=0.05, windowStepTime=0.04),
		'ESC50': self.param_func(nFFT=512, windowLenTime=0.025, windowStepTime=0.01),
		'FSC': self.param_func(nFFT=512, windowLenTime=0.025, windowStepTime=0.01),
		'Spatial': self.param_func(nFFT=512, windowLenTime=0.025, windowStepTime=0.01),
		'Synthetic': self.param_func(nFFT=512, windowLenTime=0.025, windowStepTime=0.01),
		}
		self.fs = None  # audio sampling rate, 16000 for GoogleCommand, NSynth, UrbanSound
		self.words = {} # dict for storing raw audio signal
		self.env_type = os.path.split(self.config.envFolder)[0]

		if len(self.env_type)==0: self.env_type=self.config.envFolder
		self.counter=0


	def loadData(self): # it will be called in sheme_vec_env or dummy_vec_env
		if self.env_type == 'pybullet':
			for i in range(self.config.taskNum):
				self.words[i] = {}

			for dataset in self.config.soundSource['dataset']:
				if dataset == 'FSC':
					self.loadFSCData_pybullet()
				else:
					self.loadSoundData_pybullet(datasetName=dataset)

		elif self.env_type == 'ai2thor':
			self.audioDataFrame = {}  # a dict containing metadata for self.words
			self.transcription={}
			self.loadFSCData_ai2thor(loadSize=self.config.soundSource['size'])

		else:
			raise NotImplementedError

		print("Sound Loaded")


	def loadFSCData_ai2thor(self, loadSize=-1):
		"""
		This function loads fluent speech dataset (FSC) for ai2thor env
		"""
		soundSource=self.config.soundSource
		df = pd.read_csv(os.path.join(self.config.commonMediaPath, 'FSC', 'data', soundSource['FSC_csv']))
		# filter objects
		objs = soundSource['FSC_obj_act'].keys()
		df = df[df.object.isin(objs)]

		locs = soundSource['FSC_locations']
		for loc in locs:
			loc_df = df[df.location.isin([loc])]  # filter location
			self.audioDataFrame[loc] = {}
			self.transcription[loc]={}
			self.words[loc]={}
			for obj in objs:  # for each object
				obj_df = loc_df[loc_df.object == obj]
				if not obj_df.empty:
					possible_act = soundSource['FSC_obj_act'][obj]
					self.audioDataFrame[loc][obj] = {}
					self.transcription[loc][obj]={}
					self.words[loc][obj] = {}
					for act in possible_act:
						self.audioDataFrame[loc][obj][act] = obj_df[obj_df.action == act]
						self.words[loc][obj][act]=[]
						self.transcription[loc][obj][act]=[]
						path_list=self.audioDataFrame[loc][obj][act]['path'].tolist()
						trans_list=self.audioDataFrame[loc][obj][act]['transcription'].tolist()
						idx=np.arange(len(path_list))
						for i in idx:
							self.fs, x = wavfile.read(os.path.join(self.config.commonMediaPath, 'FSC', path_list[i]))
							if x.size/self.fs>self.config.soundSource['FSC_max_sound_dur']:
								continue
							self.words[loc][obj][act].append(x)
							self.transcription[loc][obj][act].append(trans_list[i])
							if len(self.words[loc][obj][act])>=loadSize:
								break

	def load2Words(self, path_list, idx, datasetName, max_sound_dur, loadSize):
		"""
		read wav files from disk according to path_list and store them to self.words
		:return:
		"""
		for j in range(len(path_list)):
			self.fs, x = wavfile.read(path_list[j])
			if x.size / self.fs > max_sound_dur:
				continue
			self.words[idx][datasetName].append(x)
			if len(self.words[idx][datasetName]) >= loadSize:
				break

	def loadFSCData_pybullet(self):
		soundSource = self.config.soundSource
		df = pd.read_csv(os.path.join(self.config.commonMediaPath, 'FSC', 'data', soundSource['FSC_csv']))
		intent_list=soundSource['items']['FSC']
		for i, item in enumerate(intent_list):
			if item is not None:
				loadSize = soundSource['size']['FSC'][i]
				loc, obj, act=item.split('_')
				assert 'FSC' not in self.words[i]
				self.words[i]['FSC']=[]
				subdf = df[(df.object==obj) & (df.action==act) & (df.location==loc)]
				path_list=(os.path.join(self.config.commonMediaPath, 'FSC')+os.sep+subdf['path']).tolist()
				self.load2Words(path_list, i, 'FSC', self.config.soundSource['max_sound_dur']['FSC'], loadSize)



	def loadSoundData_pybullet(self, datasetName):
		"""
		This function loads Google Command Dataset, NSynth Dataset, and UrbanSound Dataset, ESC-50 Dataset for PyBullet env
		"""
		soundSource = self.config.soundSource
		word_dir = os.path.join(self.config.commonMediaPath, datasetName, self.soundSource['train_test'])
		assert os.path.isdir(word_dir)
		intent_list = soundSource['items'][datasetName]
		for i, item in enumerate(intent_list):
			if item is not None:
				loadSize = soundSource['size'][datasetName][i]
				assert datasetName not in self.words[i]
				self.words[i][datasetName] = []
				folderPath = os.path.join(word_dir, item)
				path_list=glob.glob(os.path.join(folderPath, '*.wav'))
				self.load2Words(path_list, i, datasetName, self.config.soundSource['max_sound_dur'][datasetName], loadSize)

	def get_mfcc(self, audioSamples, param, mfcc_from):
		# calculate mfcc at run time to reduce memory usage
		if mfcc_from=='torchaudio': # use mfcc function provided by torchaudio
			mfcc_func = torch_mfcc(sample_rate=self.fs, n_mfcc=40, log_mels=True,
							 melkwargs={"n_fft": param.nFFT, "win_length": int(param.windowLenTime*self.fs),
										"hop_length": int(param.windowStepTime*self.fs),
										"n_mels": 40, "f_min":0, "f_max":None, 'window_fn':torch.hamming_window}, )
			if audioSamples.dtype==np.int16: # torchaudio requires float32
				audioSamples = (audioSamples / 32768.).astype(np.float32)  # normalize to 32-bit float
			sound_feat=mfcc_func(torch.from_numpy(audioSamples))
			sound_feat=torch.transpose(sound_feat, dim0=0, dim1=1).numpy()
		else:
			sound_feat = mfcc(audioSamples, self.fs, winlen=param.windowLenTime,
							  winstep=param.windowStepTime,
							  numcep=40, nfilt=40, nfft=param.nFFT, winfunc=np.hamming)

		sound_feat = self.processSoundFeat(sound_feat)
		return sound_feat

	def getAudioSamples(self, intentIdx, rand_fn, trans_fn=None):
		"""
		get an audio sample from self.words and transform it with trans_fn
		:return: the transformed audio and the parameters for audio features
		"""
		if intentIdx>self.config.taskNum-1:
			intentIdx=self.config.taskNum-1

		possible_dataset_list=list(self.words[intentIdx].keys())
		datasetChosen=possible_dataset_list[rand_fn(0, len(possible_dataset_list), size=())]
		soundIndx = rand_fn(0, len(self.words[intentIdx][datasetChosen]), size=())
		audioSamples = self.words[intentIdx][datasetChosen][soundIndx] # audio sample in 16-bit signed int

		if trans_fn is not None:
			audioSamples = (audioSamples / 32768.).astype(np.float32) # normalize to 32-bit float
			audioSamples=trans_fn(audioSamples, self.fs)

		param = self.param_dict[datasetChosen]

		return audioSamples, param

	def genSoundFeat(self, intentIdx, featType, rand_fn, mfcc_from='torchaudio', trans_fn=None):
		"""
		generate sound feature according to intentIdx. Each intent is associate with an intentIdx
		it is the case for pybullet env
		"""
		audioSamples, param=self.getAudioSamples(intentIdx, rand_fn, trans_fn)

		if featType == 'MFCC':
			sound_feat=self.get_mfcc(audioSamples, param, mfcc_from)
		else:
			raise NotImplementedError

		return sound_feat, audioSamples



	def genSoundFeatFromTask(self, task, featType, mfcc_from=None,rand_fn=None):
		"""
		generate sound feature according to task. A task is a struct with attributes loc, obj, and act
		it is the case for ai2thor env
		"""
		soundList=self.words[task.loc][task.obj][task.act]
		soundIndx=rand_fn(0, len(soundList), size=())
		audioSamples=soundList[soundIndx]
		audioTranscription=self.transcription[task.loc][task.obj][task.act][soundIndx]


		if featType == 'MFCC':
			param=self.param_dict[self.config.soundSource['dataset']]
			sound_feat=self.get_mfcc(audioSamples, param, mfcc_from)

		else:
			raise NotImplementedError

		return sound_feat, audioSamples, audioTranscription

	def getAudioFromTask(self, random_func, tsk, Task, trans_fn=None):
		idx=random_func.randint(low=0, high=len(self.config.synonym[tsk.loc]), size=())
		loc = self.config.synonym[tsk.loc][idx]

		idx = random_func.randint(low=0, high=len(self.config.synonym[tsk.obj]), size=())
		obj = self.config.synonym[tsk.obj][idx]

		obj_act = self.config.soundSource['FSC_obj_act'][obj]
		synonym_act = self.config.synonym[tsk.act]
		act = list(set(obj_act).intersection(synonym_act))[0]

		sound_feat, audioSamples, audioTranscription = self.genSoundFeatFromTask(task=Task(loc, obj, act),
																				 featType='MFCC',
																				 rand_fn=random_func.randint)
		return sound_feat, audioSamples, audioTranscription



	def processSoundFeat(self, sound_feat):
		sound_feat = np.expand_dims(sound_feat, axis=0)
		# process the sound
		nf = sound_feat.shape[1]
		if self.config.sound_dim[1] < nf:  # drop extra if the length is too long
			sound_feat = sound_feat[:, :self.config.sound_dim[1], :]
		else:  # pad 0 if the length is not long enough
			zeroPadShape = list(self.config.sound_dim)
			zeroPadShape[1] = self.config.sound_dim[1] - nf
			sound_feat = np.concatenate((sound_feat, np.zeros(zeroPadShape)), axis=1)

		return sound_feat



