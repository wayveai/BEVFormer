import os
import torch
import matplotlib.pyplot as plt

from mmcv import Config
from mmcv.parallel import MMDataParallel

from projects.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

model_size = 'small'

cfg = Config.fromfile(f'projects/configs/bevformer/bevformer_{model_size}.py')


# build the dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=True,
    shuffle=False,
    nonshuffler_sampler=cfg.data.nonshuffler_sampler,
)


# build the model
args = cfg.model.copy()
args.pop('type')
# print(args)
model = BEVFormer(**args)
model.load_state_dict(torch.load(f'ckpts/bevformer_{model_size}_epoch_24.pth')['state_dict'])
model = MMDataParallel(model) # necessary for model to handle nested data structure from dataloader
model.eval()


# run inference
for i, data in enumerate(data_loader):
    with torch.no_grad():
        # print(data['img_metas'])
        # print(data['img'])

        result = model(return_loss=False, rescale=True, **data)


        result = result[0]['pts_bbox']
        boxes = result['boxes_3d']
        corners = boxes.corners
        labels = result['labels_3d']
        scores = result['scores_3d']

        # project boxes to images
        lidar2img = torch.tensor(data['img_metas'][0].data[0][0]['lidar2img'][0], dtype=torch.float32)
        corners_img = (lidar2img[:3,:3] @ corners.unsqueeze(-1) + lidar2img[:3,3:]).squeeze(-1)
        corners_cam = corners_img[...,:2] / corners_img[...,2:]
        masks = (corners_img[...,2]>0).all(1)

        img_norm_cfg = data['img_metas'][0].data[0][0]['img_norm_cfg']
        img = data['img'][0].data[0]
        img = img[0,0].permute(1,2,0) * img_norm_cfg['std'] + img_norm_cfg['mean']

        # plot
        plt.subplot(1,2,1)
        inds = [0,4,7,3,0]
        for corner, label, score in zip(corners, labels, scores):
            if score > .15:
                plt.plot(corner[inds,0], corner[inds,1], '-')
        plt.axis('equal')
        
        plt.subplot(1,2,2)
        plt.imshow(img/255)
        inds = [0,1,2,3,0, 4,5,6,7,4, 5,1,2,6,7,3]
        for corner, label, score, mask in zip(corners_cam, labels, scores, masks):
            if mask and score > .15:
                plt.plot(corner[inds,0], corner[inds,1], '-')
        plt.axis([0,img.shape[1],img.shape[0],0])
        plt.show()
