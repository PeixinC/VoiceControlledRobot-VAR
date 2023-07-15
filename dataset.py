from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import ConcatDataset
import pickle
import glob
import torch
import os
import numpy as np


class VARDataset(Dataset):
    def __init__(self, picklePath, config, **kwargs):
        self.filePath=picklePath
        self.config=config
        with open(self.filePath, 'rb') as f:
            self.ground_truth_pair = pickle.load(f)
        self.audio = kwargs['audio']

        if config.name == 'AI2ThorConfig':
            from Envs.ai2thor.RL_env_VAR import Task
            self.Task = Task

            # task list
            self.tl = []  # taskList

            for loc in self.config.allTasks:
                for obj in self.config.allTasks[loc]:
                    for act in self.config.allTasks[loc][obj]:
                        t = Task(loc=loc, obj=obj, act=act)
                        self.tl.append(t)

        return


    def getImgSoundPair(self, gt, sn_id):
        sound_positive = None
        trans_fn = None
        if gt == self.config.taskNum:
            sound_positive = np.zeros(shape=self.config.sound_dim)
            if self.config.name == 'AI2ThorConfig':
                neg_tsk = self.tl[sn_id]
                sound_negative, negaitve_audio, _ = self.audio.getAudioFromTask(torch, neg_tsk, self.Task, trans_fn=trans_fn)
            else:
                sound_negative, _ = self.audio.genSoundFeat(intentIdx=sn_id, featType='MFCC', rand_fn=torch.randint, trans_fn=trans_fn)

        else:
            if self.config.name == 'AI2ThorConfig':
                pos_tsk = self.tl[gt]
                sound_positive, positive_audio, _ = self.audio.getAudioFromTask(torch, pos_tsk, self.Task, trans_fn=trans_fn)

                if sn_id == self.config.taskNum:
                    sound_negative = np.zeros(shape=self.config.sound_dim)
                else:
                    neg_tsk = self.tl[sn_id]
                    sound_negative, negaitve_audio, _ = self.audio.getAudioFromTask(torch, neg_tsk, self.Task, trans_fn=trans_fn)
            else:
                sound_positive, _ = self.audio.genSoundFeat(intentIdx=gt, featType='MFCC', rand_fn=torch.randint, trans_fn=trans_fn)

                if sn_id == self.config.taskNum:
                    sound_negative = np.zeros(shape=self.config.sound_dim)
                else:
                    sound_negative, _ = self.audio.genSoundFeat(intentIdx=sn_id, featType='MFCC', rand_fn=torch.randint, trans_fn=trans_fn)
        return sound_positive, sound_negative

    def __getitem__(self, index):
        # assume the channel of the image in the dataset is the first dimension
        # assume the sound has shape (1, frame, features)
        image=torch.from_numpy(self.ground_truth_pair[index]['image'])
        image=(image/255.).float()

        # choose audio according to ground_truth
        gt = int(self.ground_truth_pair[index]['ground_truth'])
        if 'sound_negative' not in self.ground_truth_pair[index]:
            if 'sound_negative_id' in self.ground_truth_pair[index]:
                sn_id = int(self.ground_truth_pair[index]['sound_negative_id'])
            else:
                sn_id = torch.randint(low=0, high=self.config.taskNum, size=()).item()
                if gt==sn_id:
                    sn_id=self.config.taskNum

            sound_positive, sound_negative=self.getImgSoundPair(gt, sn_id)
        else:
            sound_positive = self.ground_truth_pair[index]['sound_positive']
            sound_negative = self.ground_truth_pair[index]['sound_negative']


        return image, \
               sound_positive,\
               sound_negative,\
               gt

    def __len__(self):
        return len(self.ground_truth_pair)

class VARFineTuneDataset(VARDataset):
    """
    During the fine-tuning, we don't have labels, so we cannot randomly associate an image with a sound
    Instead, we will associate an image with a sound one time in __init__ and this association will not change
    during the training
    """
    def __init__(self, picklePath, config, **kwargs):
        VARDataset.__init__(self, picklePath, config, **kwargs)

        for item in self.ground_truth_pair:
            gt = int(item['ground_truth'])

            if 'sound_negative' not in item:
                if 'sound_negative_id' in item:
                    sn_id = int(item['sound_negative_id'])
                else:
                    sn_id = torch.randint(low=0, high=self.config.taskNum, size=()).item()
                    if gt == sn_id:
                        sn_id = self.config.taskNum

                sound_positive, sound_negative=self.getImgSoundPair(gt, sn_id)
                item['sound_positive']=sound_positive
                item['sound_negative']=sound_negative
            else: # assume sound_positive and sound_negative are provided
                pass


    def __getitem__(self, index):
        # assume the channel of the image in the dataset is the first dimension
        image = torch.from_numpy(self.ground_truth_pair[index]['image'])
        image = (image / 255.).float()

        gt = int(self.ground_truth_pair[index]['ground_truth'])
        sound_positive = self.ground_truth_pair[index]['sound_positive']
        sound_negative = self.ground_truth_pair[index]['sound_negative']

        return image, \
               sound_positive, \
               sound_negative, \
               gt


def loadEnvData(data_dir, config, batch_size, shuffle, num_workers, drop_last, loadNum=None,
                dtype=VARDataset, train_test='train'):
    # load audio dataset
    from Envs.audioLoader import audioLoader
    audio = audioLoader(config=config)
    audio.loadData()
    all_datasets = []
    for i,dirs in enumerate(data_dir):
        assert os.path.exists(dirs)
        path=os.path.join(dirs, train_test)
        if loadNum is None or loadNum[i]=='all':
            for filePath in glob.glob(os.path.join(path, '*.pickle')):
                all_datasets.append(dtype(picklePath=filePath, config=config, audio=audio))
        else:
            fileList=glob.glob(os.path.join(path, '*.pickle'))
            if len(fileList)>int(loadNum[i]):
                fileList=np.random.choice(fileList, size=int(loadNum[i]))
            for filePath in fileList:
                all_datasets.append(dtype(picklePath=str(filePath), config=config, audio=audio))

    final_dataset = ConcatDataset(all_datasets)
    generator = torch.utils.data.DataLoader(final_dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            drop_last=drop_last)
    num=[0]*(config.taskNum+1)
    for dataset in final_dataset.datasets:
        for pairs in dataset.ground_truth_pair:
            num[int(pairs['ground_truth'])]=num[int(pairs['ground_truth'])]+1
    print("The number of pairs for each object in the dataset is:", num)
    return generator, final_dataset
