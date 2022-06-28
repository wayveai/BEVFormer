import os
import torch
import matplotlib.pyplot as plt

from mmcv import Config
from mmcv.parallel import MMDataParallel

from projects.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


cfg = Config.fromfile('projects/configs/bevformer/bevformer_tiny.py')


# from mmdet3d.models import build_model
# from mmdet3d.models import build_detector
# from mmdet.models.builder import MODELS as MMDET_DETECTORS

# from mmcv.cnn import MODELS as MMCV_MODELS
# from mmcv.utils import Registry

# # import modules from plguin/xx, registry will be updated
# if hasattr(cfg, 'plugin'):
#     if cfg.plugin:
#         import importlib
#         if hasattr(cfg, 'plugin_dir'):
#             plugin_dir = cfg.plugin_dir
#             _module_dir = os.path.dirname(plugin_dir)
#             _module_dir = _module_dir.split('/')
#             _module_path = _module_dir[0]

#             for m in _module_dir[1:]:
#                 _module_path = _module_path + '.' + m
#             print(_module_path)
#             plg_lib = importlib.import_module(_module_path)
#         else:
#             # import dir is the dirpath for the config file
#             _module_dir = os.path.dirname(args.config)
#             _module_dir = _module_dir.split('/')
#             _module_path = _module_dir[0]
#             for m in _module_dir[1:]:
#                 _module_path = _module_path + '.' + m
#             print(_module_path)
#             plg_lib = importlib.import_module(_module_path)

# model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
# model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
# model = MMDET_DETECTORS.build(cfg.model, default_args=dict(train_cfg=None, test_cfg=cfg.get('test_cfg')))



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
model.load_state_dict(torch.load('ckpts/bevformer_tiny_epoch_24.pth')['state_dict'])
model = MMDataParallel(model) # necessary for model to handle nested data structure from dataloader
model.eval()


# run inference
for i, data in enumerate(data_loader):
    with torch.no_grad():
        # print(data['img_metas'])
        # print(data['img'])

        result = model(return_loss=False, rescale=True, **data)
        print(result)

        lidar2img = data['img_metas'][0].data[0][0]['lidar2img'][0]
        print(lidar2img)

        img_norm_cfg = data['img_metas'][0].data[0][0]['img_norm_cfg']
        img = data['img'][0].data[0]
        img = img[0,0].permute(1,2,0) * img_norm_cfg['std'] + img_norm_cfg['mean']
        plt.imshow(img/255)
        plt.show()

        break
