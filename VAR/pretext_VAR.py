from tqdm import trange
import torch
import numpy as np
import os
from utils import get_scheduler
from dataset import loadEnvData
import torch.optim as optim
from cfg import main_config
import pandas as pd
from pretext import Pretext

class VAR_Pretext(Pretext):
	def __init__(self):
		super().__init__(main_config())
		
	def trainRepresentation(self, epoch, lr, start_ep=0, plot=False):
		print('Begin representation training')
		# load data
		data_generator, ds = loadEnvData(data_dir=self.config.pretextDataDir,
										 config=self.config,
										 batch_size=self.config.pretextTrainBatchSize,
										 shuffle=True,
										 num_workers=self.config.pretextDataNumWorkers, # change it to 0 if hangs
										 drop_last=False,
										 loadNum=self.config.pretextDataFileLoadNum,
										 dtype=self.config.pretextDataset)
	
		if not os.path.isdir(self.config.pretextModelSaveDir):
			os.makedirs(self.config.pretextModelSaveDir)
	
		self.pretextModel.train()
	
		optimizer = optim.Adam(filter(lambda parameters: parameters.requires_grad, self.pretextModel.parameters()),
							   lr=lr,
							   weight_decay=self.config.pretextAdamL2)
	
		scheduler = get_scheduler(self.config, optimizer)
		criterion = torch.nn.TripletMarginLoss(margin=self.config.tripletMargin, p=2)
		norm_criterion = torch.nn.BCEWithLogitsLoss()
	
		loss_list=[]
	
		# main training loop
		for ep in trange(epoch, position=0):
			if self.config.plotRepresentation >= 0 and ep % self.config.plotRepresentation == 0 and ep > 0 and plot:
				self.pretextModel.eval()
				self.plotRepresentation(data_generator)
				self.pretextModel.train()
	
			loss_ep = []
			loss_ep_img_norm = []
			loss_ep_sound_norm = []

	
			for n_iter, (image, sound_positive, sound_negative, gt) in enumerate(data_generator):
				self.pretextModel.zero_grad()
				optimizer.zero_grad()
				d  = self.pretextModel(image.to(self.device), sound_positive.float().to(self.device),
									   sound_negative.float().to(self.device))
				image_feat = d['image_feat']
				sound_feat_positive = d['sound_feat_positive']
				sound_feat_negative=d['sound_feat_negative']

				loss_triplet = criterion(image_feat, sound_feat_positive, sound_feat_negative)

				loss = 1.0 * loss_triplet
	
				loss.backward()
				optimizer.step()
				loss_ep.append(loss_triplet.item())
	
			if self.config.pretextLRStep == "step":
				scheduler.step()
	
			if (ep + 1) % self.config.pretextModelSaveInterval == 0 or ep+1==epoch:
				fname = os.path.join(self.config.pretextModelSaveDir, str(start_ep+ep) + '.pt')
				if not os.path.exists(self.config.pretextModelSaveDir):
					os.makedirs(self.config.pretextModelSaveDir)
				torch.save(self.pretextModel.state_dict(), fname, _use_new_zipfile_serialization=False)
				print('Model saved to ' + fname)
	
			avg_loss = np.sum(loss_ep) / len(loss_ep)
			loss_list.append(avg_loss)

			print('average loss', avg_loss)

		if self.config.pretextTrain:
			df = pd.DataFrame({'avg_loss': loss_list})
			save_path = os.path.join(self.config.pretextModelSaveDir, 'progress.csv')
			df.to_csv(save_path, mode='w', header=True, index=False)
			print('results saved to', save_path)
		print('Pretext Training Complete')
		self.pretextModel.eval()
		if self.config.plotRepresentation >= 0 and plot:
			self.plotRepresentation(data_generator)

