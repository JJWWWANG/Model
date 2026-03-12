"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import os

from einops import rearrange, repeat
from torch import nn, einsum
import math
import torch
from torchvision import models
from .loss import CCCLoss,AULoss,FocalLoss_Ori
from torch.functional import F
import numpy as np
from collections import OrderedDict
from .vformer import VideoModel
from .audio import AudioModel
from .heads import AU_former
from .vformer import TFormer


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input

def load_pretrain(model,weight_path):
    print('Loading former weight')
    pretrained_dict = torch.load(weight_path,map_location='cpu')
    new_state_dict=OrderedDict()
    for k,v in pretrained_dict.items():
        new_name = k.replace('module.','')
        new_state_dict[new_name]=v
    model.load_state_dict(new_state_dict,strict=False)

class AudioFormer(nn.Module):
    def __init__(self, modality='A', audio_pretrained=False, task='EX'):
        super(AudioFormer, self).__init__()
        self.audio_model = AudioModel(pretrained=audio_pretrained)
        #self.audio_model = AudioFTDNNModel(pretrained=audio_pretrained)
        self.task = task
        # self.fc_video = nn.Sequential(nn.Dropout(0.0),
        #                               nn.Linear(in_features=self.video_model.r2plus1d.fc._modules['1'].in_features,
        #                                         out_features=12 + 8 + 2))
        self.modes = ['audio_features']
        self.audio_model.resnet.fc = Dummy()
        self.au_head = AU_former(dropout=0.2)

    def forward(self, x):
        #print(audio.shape) #[64, 1, 64, 1001]
        audio_model_features = self.audio_model(x)
        au_out,transfomer_features = self.au_head(audio_model_features)

        return transfomer_features

class VisualFormer(nn.Module):
    def __init__(self, modality='A;V', video_pretrained = True, task='EX', num_patches=16):
        super(VisualFormer, self).__init__()

        self.video_model = VideoModel(num_patches=num_patches)
        self.video_model.config_modality(modality)
        self.task = task
        self.modes = ["clip"]
        self.au_head = AU_former(input_dim=self.video_model.fc.in_features)
        self.video_model.fc = Dummy()

    def forward(self, x):
        video_model_features = self.video_model(x)
        au_out,transfomer_features = self.au_head(video_model_features)
        return transfomer_features

class TwoStreamAuralVisualFormer(nn.Module):
    def __init__(self, modality='A;V;M', video_pretrained=True, audio_pretrained=True, task='EX', num_patches=16):
        super(TwoStreamAuralVisualFormer, self).__init__()
        self.audio_model = AudioFormer()
        self.video_model = VisualFormer(num_patches=num_patches)
        if video_pretrained:
            load_pretrain(self.video_model,r'K:\ABAW2022\models\pretrain\vformer.pth')
            for p in self.video_model.parameters():
                p.requires_grad = False
        if audio_pretrained:
            load_pretrain(self.audio_model,r'K:\ABAW2022\models\pretrain\audio.pth')
            for p in self.audio_model.parameters():
                p.requires_grad = False
        self.task = task
        self.t_former = TFormer(num_patches=12, dim=256)
        self.au_head = AU_former(input_dim=256, dropout=0.2)
        self.modes = ['clip', 'audio_features']
        self.loss_EX = FocalLoss_Ori(num_class=7, gamma=2.0, ignore_index=7, reduction='mean')
        self.loss_AU = AULoss()
        self.loss_VA = CCCLoss() #nn.MSELoss()

    def forward(self, x):
        audio = x['audio_features']
        clip = x['clip']

        # mp = r'/home/qitam/sdb2/home/qiteam_project/lhc/project/ABAW3/Multi-modal-Multi-label-Facial-Action-Unit-Detection-with-Transformer-main/results/inputs'
        #
        # def nom(array):
        #     return (array - array.min())/(array.max()-array.min())*255
        #
        # import cv2
        # import matplotlib.pyplot as plt
        # img = x['origin'].cpu().numpy()
        # au = x['audio'].cpu().numpy()
        # a = audio.cpu().numpy()
        # # a = np.uint8(nom(a))
        # # a = np.transpose(a, (0, 2, 3, 1))
        # for i in range(len(img)):
        #     mpi = mp + f'/{i}'
        #     os.makedirs(mpi, exist_ok=True)
        #     # cv2.imwrite(mpi + '/mel.png', a[i])
        #     plt.imshow(a[i, 0])
        #     plt.title('Mel Spectrogram')
        #     plt.savefig(mpi + '/mel.png')
        #     plt.close()
        #     plt.plot(au[i, 0])
        #     plt.title('Audio Signal')
        #     plt.xticks([0, 88200, 176400, 264600, 352800, 441000], [0, 2, 4, 6, 8, 10])
        #     plt.xlabel('Time/s')
        #     plt.ylabel('Amplitude')
        #     plt.savefig(mpi + '/audio.png')
        #     plt.close()
        #     for j in range(len(img[i])):
        #         cv2.imwrite(mpi+f'/img{j}.png', img[i, j])

        audio_model_features = self.audio_model(audio)
        video_model_features = self.video_model(clip)

        features = torch.cat([audio_model_features, video_model_features], dim=2)
        bs = features.shape[0]
        out = torch.zeros(bs, 21).cuda()
        if self.task == 'AU':
            features = self.t_former(features)
            au_out, _ = self.au_head(features)
            out[:,:12] = au_out
        return out

    def get_ex_loss(self, y_pred, y_true):
        y_pred = y_pred[:, 12:19]
        y_true = y_true.view(-1)
        loss = self.loss_EX(y_pred, y_true)
        return loss

    def get_au_loss(self, y_pred, y_true):
        #y_pred = torch.sigmoid(y_pred[:, :12])
        loss = self.loss_AU(y_pred[:, :12], y_true)
        return loss

    def get_va_loss(self, y_pred, y_true):
        y_pred_v = torch.tanh(y_pred[:, 19])
        y_pred_a = torch.tanh(y_pred[:, 20])
        loss = 2*self.loss_VA(y_pred_v, y_true[:, 0]) + self.loss_VA(y_pred_a, y_true[:, 1])
        return loss

    def get_mt_loss(self, y_pred, y_true, normalize=False):  # multi-task loss
        loss_ex = self.get_ex_loss(y_pred, y_true['EX'])
        loss_au = self.get_au_loss(y_pred, y_true['AU'])
        loss_va = self.get_va_loss(y_pred, y_true['VA'])
        if normalize:
            valid_ex_label_num = np.sum(y_true['EX'].detach().cpu().numpy() != 7)
            if valid_ex_label_num != 0:
                loss_ex = loss_ex / valid_ex_label_num
            else:
                device = y_true.device
                loss_ex = torch.tensor(0.0, requires_grad=True).to(device)

            valid_au_label_num = np.sum((y_true['AU'].detach().cpu().numpy() != -1))
            if valid_au_label_num != 0:
                loss_au = loss_au / valid_au_label_num
            else:
                device = y_true.device
                loss_au = torch.tensor(0.0, requires_grad=True).to(device)

            valid_va_label_num = np.sum(y_true['VA'].detach().cpu().numpy() != -5.0)
            if valid_va_label_num != 0:
                loss_va = loss_va / valid_va_label_num
            else:
                device = y_true.device
                loss_va = torch.tensor(0.0, requires_grad=True).to(device)

        return [loss_ex, loss_au, loss_va]
