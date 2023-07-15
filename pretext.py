import os
import time
from tqdm import trange
import pickle
import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from dataset import loadEnvData
import torch.optim as optim
from utils import get_scheduler, confirm_from_user, drawArrows
import warnings
from cfg import gym_register, main_config, ENV, TASK
from shutil import copyfile
import torch.backends.cudnn as cudnn
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg')


class Pretext(object):
    def __init__(self, config):
        self.config=config

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.pretextModel = None  # no need to load pretext model for now


    def collectPretextData(self, fileName=None):
        print("Begin collecting...")
        targetNum = self.config.pretextCollectNum
        collectedNum = [0] * (self.config.taskNum + 1)

        # create parallel Envs
        from Envs.vec_env.envs import make_vec_envs
        envs = make_vec_envs(env_name=self.config.pretextEnvName,
                             seed=self.config.pretextEnvSeed,
                             num_processes=self.config.pretextNumEnvs,
                             gamma=None,
                             device=None,
                             randomCollect=True,
                             config=self.config)

        # collect data for pretext training
        observations = []
        _ = envs.reset()
        observation = envs.unwrapped.obs_list
        for pairs in observation:
            if collectedNum[int(pairs['ground_truth'])] < targetNum[int(pairs['ground_truth'])]:
                observations = observations + [copy.deepcopy(pairs)]
                collectedNum[int(pairs['ground_truth'])] = collectedNum[int(pairs['ground_truth'])] + 1
        epoch = 0
        while epoch <= self.config.pretextDataNumFiles:
            if epoch == self.config.pretextDataNumFiles and sum(collectedNum) < sum(targetNum):
                self.config.pretextDataNumFiles = self.config.pretextDataNumFiles + 3
                print('Increase number of files')
            print("Number of pairs for each object", collectedNum)
            for episode in trange(self.config.pretextDataEpisode, position=0):

                for i in range(self.config.pretextEnvMaxSteps):
                    if self.config.render:

                        envs.render()
                        if not self.config.pretextManualControl:
                            time.sleep(2)
                    action = [0] * self.config.pretextNumEnvs  # dummy action. True random action is decided in env
                    _, _, _, _ = envs.step(action)
                    observation = envs.unwrapped.obs_list

                    for pairs in observation:
                        if collectedNum[int(pairs['ground_truth'])] < targetNum[int(pairs['ground_truth'])]:
                            observations = observations + [copy.deepcopy(pairs)]
                            collectedNum[int(pairs['ground_truth'])] = collectedNum[int(pairs['ground_truth'])] + 1


                if sum(collectedNum) == sum(targetNum):
                    break

            # save observations as pickle files
            # observations is a list of dict [{'image':, 'sound_positive':, 'sound_negative':, 'ground_truth':}, ...]
            filePath = os.path.join(self.config.pretextDataDir[0], 'train')
            if not os.path.isdir(filePath):
                os.makedirs(filePath)
            if fileName is None:
                filePath = os.path.join(filePath, 'data_' + str(epoch) + '.pickle')
            else:
                filePath = os.path.join(filePath, fileName + '.pickle')
            with open(filePath, 'wb') as f:
                pickle.dump(observations, f, protocol=pickle.HIGHEST_PROTOCOL)
            observations.clear()

            if sum(collectedNum) == sum(targetNum):
                break

            epoch = epoch + 1

        envs.close()
        return epoch

    def loadPretextModel(self):
        """
        load pretextModel from config.pretextModelLoadDir and send the model to self.device
        :return: None
        """
        weight_path = self.config.pretextModelLoadDir
        if self.pretextModel is None: self.pretextModel = self.config.pretextModel(self.config)
        self.pretextModel.load_state_dict(torch.load(weight_path))
        self.pretextModel.to(self.device).eval()
        print('Load weights for pretextModel from', weight_path)

    def manuallyCollectPretextData(self):
        from Envs.vec_env.envs import make_vec_envs
        envs = make_vec_envs(env_name=self.config.pretextEnvName,
                             seed=self.config.pretextEnvSeed,
                             num_processes=1,
                             gamma=None,
                             device=None,
                             randomCollect=True,
                             config=self.config,
                             pretextObj=self)
        if self.config.realTimeVec:
            fig, ax, figText=self.initRealTimePlot()
        observation = envs.reset()
        quiver_img=None
        while True:
            envs.render()
            O, reward, done, info = envs.step([0])
            with torch.no_grad():
                d = self.pretextModel(torch.from_numpy(O['image'] / 255.).float().to(self.device), None, None)
            image_feat = d['image_feat'].to('cpu').numpy()
            if self.config.realTimeVec:
                quiver_img, _ = drawArrows(ax, fig, v_img=image_feat,
                                                                v_sound=None, quiver_img=quiver_img,
                                                                quiver_sound=None)

    def testRepresentation(self):
        device = self.device
        if self.config.pretextTestMethod=='plot':
            self.trainRepresentation(epoch=0, lr=0, start_ep=0, plot=True)
            exit()

        else:
            raise NotImplementedError

    def project2representation_with_ground_truth(self, data_generator, project_for='medoid', req_grad=False):
        """
        project all the data in data_generator through the pretextModel and store the embeddings according to
        ground-truth task id.
        :param data_generator: pytorch data_generator
        :param project_for: choose from 'plot'
        :param req_grad: if False, the pretextModel is in inference mode and will not be updated
        :return: feature point list
        """
        # setup return
        if project_for == 'plot':
            # image and sound feature with a label column
            feat_point = {'img': [], 'sound': [], 'lastBatchNum':-1}
        else:
            raise NotImplementedError


        with torch.set_grad_enabled(req_grad):
            for n, data in enumerate(data_generator):
                # parse data
                img = data[0]
                sp = data[1]
                gt = data[3]  # sound negative is data[2]


                if project_for == 'plot':
                    if n > self.config.plotNumBatch:
                        break  # show only self.config.plotNumBatch batch size data points on the plot
                    else:
                        if n == self.config.plotNumBatch and self.config.annotateLastBatch:
                            # save this batch of image to self.config.episodeImgSaveDir with ID
                            for j, pic in enumerate(img):
                                pic = np.transpose((pic.numpy() * 255).astype(np.uint8), (1, 2, 0))
                                imgSave = cv2.resize(pic,
                                                     (self.config.episodeImgSize[1], self.config.episodeImgSize[0]))
                                if self.config.episodeImgSize[2] == 3:
                                    imgSave = cv2.cvtColor(imgSave, cv2.COLOR_RGB2BGR)
                                fileName = 'lastBatch' + str(j) + '_'+str(gt[j].item())+'.jpg'
                                cv2.imwrite(os.path.join(self.config.episodeImgSaveDir, fileName), imgSave)
                            feat_point['lastBatchNum']=img.size()[0]

                        features = self.pretextModel(img.to(self.device), sp.float().to(self.device), None)

                        img_feat, sp_feat = features['image_feat'].cpu().numpy(), features['sound_feat_positive'].cpu().numpy()

                        feat_point['img'].append(np.concatenate([img_feat, gt[:,None]], axis=1))
                        feat_point['sound'].append(np.concatenate([sp_feat, gt[:,None]], axis=1))


        if project_for == 'plot':
            feat_point['img'] = np.concatenate(feat_point['img'], axis=0)
            feat_point['sound'] = np.concatenate(feat_point['sound'], axis=0)

        else:
            raise NotImplementedError

        return feat_point

    def plotRepresentation(self, generator, **kwargs):
        fig = plt.figure()
        if self.config.representationDim == 3:  # 3d scatter plot
            ax = fig.add_subplot(111, projection='3d')
            ax.set_zlabel('Z Label')
            # draw a gray sphere
            ax.plot([-1, 1], [0, 0], [0, 0], color="k", alpha=0.2, linewidth=1)
            ax.plot([0, 0], [-1, 1], [0, 0], color="k", alpha=0.2, linewidth=1)
            ax.plot([0, 0], [0, 0], [-1, 1], color="k", alpha=0.2, linewidth=1)
            ax.set_axis_off()
            u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
            x = np.cos(u) * np.sin(v)
            y = np.sin(u) * np.sin(v)
            z = np.cos(v)
            ax.plot_wireframe(x, y, z, color="lightgray", alpha=0.2, linewidth=1)

        else:  # when the dimension is not 3
            ax = fig.add_subplot(111)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        colors = ['r', 'y', 'b', 'g', 'tab:purple', 'c', 'tab:pink', 'tab:orange', 'tab:brown']

        feat_point=self.project2representation_with_ground_truth(generator, project_for='plot')
        V, A=feat_point['img'], feat_point['sound']

        if self.config.representationDim>3:# do t-SNE first
            print("Performing t-SNE...")
            VA=np.concatenate([V, A], axis=0)
            tsne = TSNE(2)
            tsne_result = tsne.fit_transform(VA[:,:-1])
            VA=np.concatenate([tsne_result, VA[:,-1, None]], axis=1)
            V=VA[:VA.shape[0]//2]
            A=VA[VA.shape[0]//2:]

        for j in range(self.config.taskNum+1):
            idx = np.where(V[:, -1] == j)[0]
            if idx.size != 0:
                img_feat = V[idx]
                sound_feat = A[idx]
                if self.config.representationDim==3:
                    ax.scatter(img_feat[:, 0], img_feat[:, 1], img_feat[:, 2], marker='o', color=colors[j], s=20, alpha=0.2)
                    ax.scatter(sound_feat[:, 0], sound_feat[:, 1], sound_feat[:, 2], marker='v', color=colors[j], s=20, alpha=0.2)

                else:
                    ax.scatter(img_feat[:, 0], img_feat[:, 1], marker='o', color=colors[j])
                    ax.scatter(sound_feat[:, 0], sound_feat[:, 1], marker='v', color=colors[j])

        if self.config.annotateLastBatch:
            V_lastBatch=V[-feat_point['lastBatchNum']:]
            for k in range(feat_point['lastBatchNum']):
                # annotate the points with index
                if self.config.representationDim == 3:
                    ax.text(V_lastBatch[k, 0], V_lastBatch[k, 1], V_lastBatch[k, 2], str(k))
                else:
                    ax.text(V_lastBatch[k, 0], V_lastBatch[k, 1], str(k))


        plt.show()
        return fig, ax


    def initRealTimePlot(self):
        """
        Initialize the real time vector plot by projecting the image and sound data from pretextDataDir
        into the representation and draw these embeddings out
        :return: matplotlib figure, axis
        """

        data_generator, ds = loadEnvData(data_dir=self.config.pretextDataDir,
                                         config=self.config,
                                         batch_size=self.config.pretextTrainBatchSize,
                                         shuffle=True,
                                         num_workers=self.config.pretextDataNumWorkers,
                                         drop_last=False,
                                         loadNum=self.config.pretextDataFileLoadNum,
                                         dtype=self.config.pretextDataset)
        plt.ion()
        fig, ax = self.plotRepresentation(data_generator)

        figText = fig.text(x=0.5, y=0.12, s="",transform=plt.gcf().transFigure, fontsize=24, ha='center', color='blue')

        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.001)

        return fig, ax, figText

    def run(self):
        cudnn.benchmark = True
        torch.manual_seed(self.config.pretextEnvSeed)
        torch.cuda.manual_seed_all(self.config.pretextEnvSeed)
        gym_register(self.config)
        if self.config.pretextCollection:

            if self.config.pretextManualCollect:
                self.loadPretextModel()
                self.manuallyCollectPretextData()
            else:
                self.collectPretextData()
            print('Data Collection Complete')

        if self.config.pretextTrain:  # if we want to train the pretext model from scratch
            self.pretextModel = self.config.pretextModel(self.config).to(self.device)
            if self.config.pretextModelFineTune: self.loadPretextModel()

            if not os.path.exists(self.config.pretextModelSaveDir):
                os.makedirs(self.config.pretextModelSaveDir)
            if ENV=='arms':
                copyfile(os.path.join('Envs', self.config.envFolder, 'tasks', TASK,  'config.py'),
                         os.path.join(self.config.pretextModelSaveDir, 'config.py'))
            else:
                copyfile(os.path.join('Envs', self.config.envFolder,  'config.py'),
                         os.path.join(self.config.pretextModelSaveDir, 'config.py'))
            p = True if self.config.plotRepresentation >= 0 else False
            self.trainRepresentation(epoch=self.config.pretextEpoch, lr=self.config.pretextLR, start_ep=0, plot=p)


        if (not self.config.pretextTrain) and (not self.config.pretextCollection):  # test
            self.loadPretextModel()
            # test our representation according to config.pretextTestMethod
            self.testRepresentation()

    def trainRepresentation(self, epoch, lr, start_ep=0, plot=False):
        raise NotImplementedError("Please Implement this method")


if __name__ == '__main__':
    mc=main_config()
    from VAR.pretext_VAR import VAR_Pretext
    var_pretext=VAR_Pretext()
    var_pretext.run()