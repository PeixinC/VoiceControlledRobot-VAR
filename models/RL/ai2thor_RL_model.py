import numpy as np
import torch.nn as nn
from models.ppo.utils import init
from ..ppo.model import NNBase, Flatten


class ai2thorNet_VAR(NNBase):
    def __init__(self, num_inputs, config=None, recurrent=False, recurrentInputSize=128, recurrentSize=128,
                 actionHiddenSize=128):
        super(ai2thorNet_VAR, self).__init__(recurrent, recurrentInputSize, recurrentSize, actionHiddenSize)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.imgCNN = nn.Sequential(
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
            Flatten()
        )


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.occupancyCNNMLP=nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.ReLU(),  # (1, 9, 9)->(32, 5, 5)
            nn.Conv2d(64, 32, 3, stride=2, padding=1), nn.ReLU(), # (32, 5, 5)->(32, 3, 3)
            Flatten(),
            nn.Linear(32*9, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU())


        self.motorMlp = nn.Sequential(
            init_(nn.Linear(3, 64)), nn.ReLU(),
            init_(nn.Linear(64, 256)), nn.ReLU(),
        )
        self.cnnMlp = nn.Sequential(
            init_(nn.Linear(128 * 3 * 3, 512)), nn.ReLU(),
            init_(nn.Linear(512, 256)), nn.ReLU())

        self.imgMotorMlp = nn.Sequential(
            init_(nn.Linear(256, 64)), nn.ReLU(),
            init_(nn.Linear(64, recurrentInputSize)), nn.ReLU(),
        )
        self.imgMotorMlp2 = nn.Sequential(
            init_(nn.Linear(recurrentSize, 256)), nn.ReLU(),
        )
        self.soundMlp = nn.Sequential(
            init_(nn.Linear(3, 128)), nn.ReLU(),
            init_(nn.Linear(128, 256)), nn.ReLU(),
            init_(nn.Linear(256, 256)), nn.ReLU(),
        )

        self.fusionMlp = nn.Sequential(
            init_(nn.Linear(256, 512)), nn.ReLU(),
            init_(nn.Linear(512, 256)), nn.ReLU(),

        )

        self.mlp_all = nn.Sequential(
            init_(nn.Linear(256, 256)), nn.ReLU(),
            init_(nn.Linear(256, 128)), nn.ReLU(),
        )

        self.actor = nn.Sequential(
            init_(nn.Linear(128, 128)), nn.ReLU(),
            init_(nn.Linear(128, actionHiddenSize)), nn.ReLU())

        self.critic = nn.Sequential(
            init_(nn.Linear(128, 128)), nn.ReLU(),
            init_(nn.Linear(128, 128)), nn.ReLU())

        self.critic_linear = init_(nn.Linear(128, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, **kwargs):
        x = inputs
        motor_imgEmb = x['image_feat']
        sound = x['goal_sound_feat']
        occupancy = x['occupancy']
        occupancy = self.occupancyCNNMLP(occupancy)
        image = x['image']
        image = self.imgCNN(image)

        image_flatten = self.cnnMlp(image)
        motor = self.motorMlp(motor_imgEmb)
        imageMotor = self.imgMotorMlp(image_flatten + motor+occupancy)

        if self.is_recurrent:
            imageMotor, rnn_hxs = self._forward_gru(imageMotor, rnn_hxs, masks)

        imageMotorRnn = self.imgMotorMlp2(imageMotor)

        sound = self.soundMlp(sound)
        fusion = sound + image_flatten

        fusion = self.fusionMlp(fusion)
        final_fusion = fusion + imageMotorRnn
        x = self.mlp_all(final_fusion)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        additional = {
        }
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs, additional