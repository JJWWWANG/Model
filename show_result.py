import warnings
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler
import numpy as np
from models import *
from dataloader import SubsetSequentialSampler, SubsetRandomSampler, Prefetcher, Aff2TestDataset, Aff2CompDataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
import os
import time
from collections import defaultdict
import opts2
import random
import logging
import matplotlib.pyplot as plt
from metrics.accf1 import MultiLabelF1_test
from torchvision.transforms import ToPILImage
import cv2

model_path = r'experiments/new/hrformer_AU_F1_MT/pretrain/best.pth' # path to the model
result_path = 'results/pic'# path where the result .txt files should be stored

# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)


# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, save_name):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 4
    grads = grads.reshape([grads.shape[0], -1])  # 5
    weights = np.mean(grads, axis=1)  # 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]  # 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    cv2.imwrite(save_name, cam_img)


if __name__ == '__main__':
    opt = opts2.parse_opt()
    opt = vars(opt)
    gpuid = 6
    opt['gpu_id'] = gpuid
    torch.cuda.set_device(gpuid)
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.enabled = True
        device = torch.device(f"cuda:{gpuid}")
    else:
        device = torch.device("cpu")
        print('cpu selected!')

    # model
    # model = VGGVisualFormer(modality='A;V', task='AU', video_pretrained=False, num_patches=1)
    model = SpatialTemporalHrFormer(modality='A;V', task='AU', video_pretrained=False, num_patches=1)
    # model = SpatialTemporalFormer(modality='A;V', task='AU', video_pretrained=False, num_patches=1)
    # model = TwoStreamAuralVisualFormer(modality='A;V', task='AU')
    modes = model.modes
    model = model.to(device)
    print('Loading weight from:{}'.format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # disable grad, set to eval
    # for p in model.parameters():
    #     p.requires_grad = False
    # for p in model.children():
    #     p.train(False)

    categories = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]
    CAM_AU_id = 10
    istransformer = True

    fmap_block = list()
    grad_block = list()

    # if istransformer:
    model.video_model.s_former.hr_emb.final_layer.register_forward_hook(farward_hook)  # 9
    model.video_model.s_former.hr_emb.final_layer.register_backward_hook(backward_hook)
    # else:
    #     model.video_model.s_former.hr_emb.register_forward_hook(farward_hook)  # 9
    #     model.video_model.s_former.hr_emb.register_backward_hook(backward_hook)
    print(f'CAM AU id: ' + categories[CAM_AU_id])
    print(f'is transformer? {istransformer}')

    # load dataset (first time this takes longer)
    # dataset = Aff2TestDataset(opt)
    dataset = Aff2CompDataset(opt)
    dataset.set_modes(modes)
    dataset.set_aug(False)

    metric_au = MultiLabelF1_test(ignore_index=-1)

    # select the frames we want to process (we choose VAL and TEST)

    downsample_rate = 10000
    downsample = np.zeros(len(dataset), dtype=int)
    downsample[np.arange(0, len(dataset) - 1, downsample_rate)] = 1
    val_sampler = SubsetSequentialSampler(np.nonzero(dataset.val_ids * downsample)[0], shuffle=False)
    print('Test set length: ' + str(sum(dataset.val_ids * downsample)))
    loader = DataLoader(dataset, batch_size=1, sampler=val_sampler, num_workers=0, pin_memory=False, drop_last=False)

    #labels = torch.zeros((len(dataset), 17), dtype=torch.float32)
    # run inference
    os.makedirs(result_path, exist_ok=True)

    to_pil_image = ToPILImage()

    bar = tqdm(loader)
    for data in bar:
        ids = data['Index'].long()
        id = ids.item()
        if id != 30000:
            continue
        result_dir = os.path.join(result_path, str(id))
        os.makedirs(result_dir, exist_ok=True)
        x = {}
        origin_img = data['origin'][0, 0]
        img = origin_img.numpy()

        ## 保存原图
        # origin_img = to_pil_image(origin_img.permute(2, 0, 1))
        # origin_img.save(os.path.join(result_dir, 'image.png'))

        for mode in modes:
            x[mode] = data[mode].to(device)

        label_ex = data['EX']
        label_ex[label_ex == -1] = 7
        labels = {
            'VA': data['VA'],
            'AU': data['AU'],
            'EX': label_ex,
        }
        result = model(x)

        predict = result.detach().cpu()
        pred_au = torch.sigmoid(predict[:, :12]).detach().cpu().squeeze().numpy()
        round_au = np.round(pred_au)
        print(pred_au)
        # 生成AU预测结果图
        plt.figure()
        plt.barh(categories[::-1], pred_au[::-1], label='HRNet+Transformer', color='lightcoral')
        plt.legend()
        plt.title("AU pred")
        plt.ylabel("AUs")
        plt.xlabel("Probability")
        plt.xlim([0, 1])
        plt.savefig(os.path.join(result_dir, 'AU pred.png'))
        plt.close()

        # # backward
        # model.zero_grad()
        # class_loss = result[0, CAM_AU_id]
        # class_loss.backward()
        #
        # # 生成cam
        # grads_val = grad_block[0].cpu().data.numpy().squeeze()
        # fmap = fmap_block[0].cpu().data.numpy().squeeze()
        #
        # # 保存cam图片
        # if istransformer:
        #     cam_show_img(img, fmap, grads_val, os.path.join(result_dir, categories[CAM_AU_id] + ' CAM Trans.png'))
        # else:
        #     cam_show_img(img, fmap, grads_val, os.path.join(result_dir, categories[CAM_AU_id] + ' CAM.png'))