import numpy as np
from os import path as osp
from pathlib import Path
from itertools import compress

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import pickle
from mmdet3d.core import show_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from projects.mmdet3d_plugin.utils import Transform

from .nuscenes_dataset import CustomNuScenesDataset


camera_order = [
    'front-forward',
    'front-right-rightward',
    'front-left-leftward',
    'back-backward',
    'back-left-leftward',
    'back-right-rightward',
]


@DATASETS.register_module()
class WayveDataset(CustomNuScenesDataset):
    CLASSES = (
        'car', 'bus', 'van', 'truck', 'bicycle', 'motorcycle', 'scooter',
        'cyclist', 'motorcyclist', 'scooterist', 'pedestrian', 'traffic_light',
        'unknown',
    )
    def __init__(
        self,
        use_vehicle_ref=False,
        queue_length=4,
        bev_size=(200, 200),
        overlap_test=False,
        *args, **kwargs,
    ):
        # If use_vehicle_ref, then the 'lidar2img' transform will actually give the transform
        # from the ground-nominal frame to the image
        self.use_vehicle_ref = use_vehicle_ref
        kwargs['queue_length'] = queue_length
        kwargs['bev_size'] = bev_size
        kwargs['overlap_test'] = overlap_test
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)

        # Make a mapping so that we can load a sample given its 'token'
        self.token_mapping = {d['token']: i for i, d in enumerate(data)}
        return data

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        can_bus = np.zeros(18)
        #  import ipdb; ipdb.set_trace()
        input_dict = dict(
            sample_idx=info['seq_num'],
            pts_filename='',
            sweeps='',
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev_frame'],
            next_idx=info['next_frame'],
            scene_token=info['timestamp_us'],
            can_bus=can_bus,
            frame_idx=info['frame_id'],
            timestamp=info['timestamp_us'] / 1e6,
            token=info['timestamp_us'],
            #  timestamp=info['timestamp_us'] / 1e6,
        )

        #  import ipdb; ipdb.set_trace()
        # G_V_L is transform from lidar to ego vehicle
        G_V_L = Transform.from_Rt(
            R.from_quat(info['lidar2ego_rotation']),
            np.array(info['lidar2ego_translation']),
        )
        G_FF_L2 = Transform.from_Rt(
            R.from_euler('xyz', (88.93, 0.008, 0.374), degrees=True),
            t=[ 0.012, -0.334, -0.408],
        )
        G_V_FF = Transform.from_Rt(
            R.from_quat(info['cameras']['front-forward']['cam2ego_rotation']),
            np.array(info['cameras']['front-forward']['cam2ego_translation']),
        )
        G_L2_L = G_FF_L2.inverse @ G_V_FF.inverse @ G_V_L

        if self.modality['use_camera']:
            #  import ipdb; ipdb.set_trace()
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type in camera_order:
                cam_info = info['cameras'][cam_type]
                image_paths.append(osp.join(self.data_root, cam_info['path']))

                # obtain lidar to image transformation matrix
                G_V_C = Transform.from_Rt(
                    R.from_quat(cam_info['cam2ego_rotation']),
                    np.array(cam_info['cam2ego_translation']),
                )
                G_C_V = G_V_C.inverse

                K = np.array(cam_info['cam_intrinsic'])
                # Make a 4x4 matrix with upper left the intrinsics
                G_im_C = Transform(K)
                cam_intrinsics.append(G_im_C.matrix)

                if self.use_vehicle_ref:
                    # Vehicle to image
                    G_im_V = G_im_C @ G_C_V
                    lidar2cam_rts.append(G_C_V)
                    lidar2img_rts.append(G_im_V.matrix)
                else:
                    G_C_L = G_V_C.inverse @ G_V_L
                    G_im_L = G_im_C @ G_C_L
                    lidar2cam_rts.append(G_C_L)
                    lidar2img_rts.append(G_im_L.matrix)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam2img=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index, info['ego2global_translation'], info['ego2global_rotation'])
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index, ego2global_t, ego2global_r):
        #  import ipdb; ipdb.set_trace()

        info = self.data_infos[index]
        cuboids = self.data_infos[index]['gt_label_boxes']['cuboids']
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = [cuboid['visibility'] != '0%' for cuboid in cuboids]
        else:
            mask = [cuboid['num_of_points'] > 0 for cuboid in cuboids]
        cuboids = list(compress(cuboids, mask))

        gt_names_3d = [cuboid['category'] for cuboid in cuboids]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # Make a cuboid tensor - these cuboids are in world coordinates
        # [x,y,z,w,l,h,yaw,vx,vy,category,confidence,unique_id]
        def encode(cuboid) -> np.ndarray:
            x, y, z = cuboid['pose']['translation']['forward_left_up']
            x_size, y_size, z_size = cuboid['size_wlh']
            yaw = -np.pi/2 - cuboid['pose']['rotation']['forward_left_up'][2]
            velocity = cuboid['velocity'] if cuboid['velocity'] is not None else [0., 0.]
            return np.array([x, y, z, x_size, y_size, z_size, yaw, *velocity])

        gt_bboxes_3d = np.stack([encode(cuboid) for cuboid in cuboids], axis=0)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        #  import ipdb; ipdb.set_trace()
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        # Translate the boxes to local position
        gt_bboxes_3d.translate(-np.array(ego2global_t))
        gt_bboxes_3d.rotate(R.from_quat(ego2global_r).inv().as_matrix())

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results
