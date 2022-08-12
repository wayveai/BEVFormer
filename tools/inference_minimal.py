import os
import torch
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from mmcv import Config
from mmcv.parallel import MMDataParallel

from projects.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


session_dir = '/mnt/remote/azure_session_dir/2d/bevformer/bevformer_small_wayve'
epoch = 7
#  cfg = Config.fromfile('projects/configs/bevformer/bevformer_small_wayve.py')
#  cfg = Config.fromfile('/home/fergal/repos/zhiqi-li/BEVFormer/work_dirs/bevformer_small_wayve_overfit/bevformer_small_wayve_overfit.py')
#  cfg = Config.fromfile('/mnt/remote/azure_session_dir/2d/bevformer/bevformer_small_wayve/bevformer_small_wayve.py')
#  cfg = Config.fromfile('/mnt/remote/azure_session_dir/2d/bevformer/bevformer_small_wayve/bevformer_small_wayve.py')
cfg = Config.fromfile(f'{session_dir}/bevformer_small_wayve.py')
cfg.data.train.ann_file = 'data/wayve/wayve_infos_temporal_train.pkl'
cfg.data.train.data_root = 'data/wayve'
cfg.data.test.ann_file = 'data/wayve/wayve_infos_temporal_train.pkl'
cfg.data.test.data_root = 'data/wayve'


#  img_norm_cfg = dict(
    #  mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
#  dataset = build_dataset(cfg.data.train)
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=0, #cfg.data.workers_per_gpu,
    dist=True,
    shuffle=False,
    nonshuffler_sampler=cfg.data.nonshuffler_sampler,
)


# build the model
args = cfg.model.copy()
args.pop('type')
# print(args)
model = BEVFormer(**args)
#  model.load_state_dict(torch.load('ckpts/bevformer_small_epoch_24.pth')['state_dict'])
#  model.load_state_dict(torch.load('ckpts/bevformer_small_epoch_24.pth')['state_dict'])
#  model.load_state_dict(torch.load(f'/mnt/remote/azure_session_dir/2d/bevformer/bevformer_small_wayve/epoch_{epoch}.pth')['state_dict'])
#  model.load_state_dict(torch.load(f'/mnt/remote/azure_session_dir/2d/bevformer/bevformer_small_wayve/epoch_{epoch}.pth')['state_dict'])
model.load_state_dict(torch.load(f'{session_dir}/epoch_{epoch}.pth')['state_dict'])
#  /mnt/remote/azure_session_dir/2d/bevformer/bevformer_small_wayve/
#  model.load_state_dict(torch.load('/home/anindya/BEVFormer/work_dirs/bevformer_small_wayve/epoch_1.pth')['state_dict'])
#  model.load_state_dict(torch.load('/home/fergal/repos/zhiqi-li/BEVFormer/work_dirs/bevformer_small_wayve_overfit/epoch_1000.pth')['state_dict'])
#  model.load_state_dict(torch.load('/home/fergal/repos/zhiqi-li/BEVFormer/work_dirs/bevformer_small_wayve_overfit/epoch_1000.pth')['state_dict'])
model = MMDataParallel(model) # necessary for model to handle nested data structure from dataloader
model.eval()
model.cuda()


# run inference
output = []
for i, data in tqdm(enumerate(data_loader)):
    if i >= 15:
        break
    with torch.no_grad():
        # print(data['img_metas'])
        # print(data['img'])

        out = {}
        #   Save the labels
        #  out['label_bbox'] = {
            #  'boxes_3d': data['gt_bboxes_3d'].data[0][0].tensor,
            #  'labels_3d': data['gt_labels_3d'].data[0][0],
        #  }
        #  del data['gt_bboxes_3d']
        #  del data['gt_labels_3d']
        #  output.append(out)
        #  continue

        result = model(return_loss=False, rescale=True, **data)

        # Save the predictions
        out['pred_bbox'] = {
            'boxes_3d': result[0]['pts_bbox']['boxes_3d'].tensor,
            'scores_3d': result[0]['pts_bbox']['scores_3d'],
            'labels_3d': result[0]['pts_bbox']['labels_3d'],
        }
        output.append(out)
        #  print(result)
        #  continue

        #  lidar2img = data['img_metas'][0].data[0][0]['lidar2img'][0]
        #  print(lidar2img)

        #  img_norm_cfg = data['img_metas'][0].data[0][0]['img_norm_cfg']
        #  img = data['img'][0].data[0]
        #  img = img[0,0].permute(1,2,0) * img_norm_cfg['std'] + img_norm_cfg['mean']
        #  plt.imshow(img/255)
        #  plt.show()

        #  break

#  with open(f'/home/fergal/pretrained.pkl', 'wb') as f:
    #  pickle.dump(output, f)
with open(f'/home/fergal/out{epoch}.pkl', 'wb') as f:
    pickle.dump(output, f)
