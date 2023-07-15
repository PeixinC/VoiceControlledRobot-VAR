import torch
import torch.nn as nn
import torch.nn.functional as F


class PretextNetBase(nn.Module):
    def __init__(self):
        super(PretextNetBase, self).__init__()

    def VAR_forward(self, image, sound_positive, sound_negative, is_train=False):
        image_feat, image_feat_raw = None, None
        sound_feat_negative = None
        pos_sound_raw = None
        image_BCE, sound_BCE = None, None

        def getSoundOutput(sound):
            raw = self.soundBranch(self, sound)
            feat = F.normalize(self.soundTriplet(raw), p=2, dim=1)
            return raw, feat

        if image is not None:
            image_feat_raw = self.imgBranch(image[:, :3, :, :])
            image_feat = F.normalize(self.imgTriplet(image_feat_raw), p=2, dim=1)

        # sound feat positive
        # at RL training and testing, we can use the cached sound encoding
        # assuming every env reset at the same time

        if sound_positive is not None and (not torch.isinf(sound_positive).all()):
            pos_sound_raw, sound_feat = getSoundOutput(sound_positive)
            self.cached_sound = sound_feat
        sound_feat_positive = self.cached_sound

        if sound_negative is not None:
            neg_rnn_out, sound_feat_negative = getSoundOutput(sound_negative)

        d = {'image_feat': image_feat, 'sound_feat_positive': sound_feat_positive,
             'sound_feat_negative': sound_feat_negative, 'image_BCE': image_BCE, 'sound_BCE': sound_BCE,
             'image_feat_raw': image_feat_raw, 'pos_sound_raw': pos_sound_raw
             }

        return d