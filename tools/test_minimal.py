from typing import Tuple, Dict, List
import json
import pickle
import os
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox


class WayveDetectionBox(DetectionBox):
    """
    A slight modification on the DetectionBox of Nuscenes. Don't want to assert on class names
    """
    def __init__(
        self,
        sample_token: str = "",
        translation: Tuple[float, float, float] = (0, 0, 0),
        size: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
        velocity: Tuple[float, float] = (0, 0),
        ego_translation: [float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
        num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
        detection_name: str = 'car',  # The class name used in the detection challenge.
        detection_score: float = -1.0,  # GT samples do not have a score.
        attribute_name: str = ''
    ):  # Box attribute. Each box can have at most 1 attribute.
        EvalBox.__init__(self, sample_token, translation, size, rotation, velocity, ego_translation, num_pts)
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name

    @classmethod
    def deserialize(cls, content):
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']) if content['velocity'] is not None else (0, 0),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   detection_name=content['detection_name'],
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                   attribute_name=content['attribute_name'])


class_ranges = {
    'car': 50,
    'bus': 50,
    'van': 50,
    'truck': 50,
    'bicycle': 40,
    'motorcycle': 40,
    'scooter': 40,
    'cyclist': 40,
    'motorcyclist': 40,
    'scooterist': 40,
    'pedestrian': 40,
    'traffic_light': 40,
}
class WayveDetectionConfig(DetectionConfig):
    def __init__(
        self,
        class_range: Dict[str, int],
        dist_fcn: str,
        dist_ths: List[float],
        dist_th_tp: float,
        min_recall: float,
        min_precision: float,
        max_boxes_per_sample: int,
        mean_ap_weight: int
    ):
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."
        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight
        self.class_names = self.class_range.keys()


class WayveDetectionEval(DetectionEval):
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.
    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.
    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.
    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.
    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 config: DetectionConfig,
                 result_path: str,
                 label_path: str,
                 output_dir: str,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.result_path = result_path
        self.label_path = label_path
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots') if self.output_dir is not None else None
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        with open(result_path, 'r') as f:
            data = json.load(f)
        self.pred_boxes = EvalBoxes.deserialize(data['results'], WayveDetectionBox)
        with open(label_path, 'r') as f:
            data = json.load(f)
        self.gt_boxes = EvalBoxes.deserialize(data['labels'], WayveDetectionBox)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(self.pred_boxes, data['egoposes'])
        self.meta = {}
        self.gt_boxes = add_center_dist(self.gt_boxes, data['egoposes'])

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_boxes(self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_boxes(self.gt_boxes, self.cfg.class_range, verbose=verbose)
        self.sample_tokens = self.gt_boxes.sample_tokens


def filter_boxes(eval_boxes, max_dist: dict, verbose: bool = True):
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):
        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        eval_boxes.boxes[sample_token] = [
            box for box in eval_boxes[sample_token]
            if box.ego_dist < max_dist[getattr(eval_boxes[sample_token][0], 'detection_name')]
        ]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        point_filter += len(eval_boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)

    return eval_boxes


def add_center_dist(eval_boxes: EvalBoxes, egoposes: dict):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    for sample_token in eval_boxes.sample_tokens:
        egopose = egoposes[sample_token]

        for box in eval_boxes[sample_token]:
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            # Note that the z component of the ego pose is 0.
            ego_translation = (box.translation[0] - egopose['ego2global_translation'][0],
                               box.translation[1] - egopose['ego2global_translation'][1],
                               box.translation[2] - egopose['ego2global_translation'][2])
            if isinstance(box, DetectionBox):
                box.ego_translation = ego_translation
            else:
                raise NotImplementedError

    return eval_boxes


if __name__ == '__main__':
    dc =  WayveDetectionConfig(
        class_ranges,
        dist_fcn='center_distance',
        dist_ths=[0.5, 1.0, 2.0, 4.0],
        dist_th_tp=2.0,
        min_recall=0.1,
        min_precision=0.1,
        max_boxes_per_sample=500,
        mean_ap_weight=5,
    )
    de = WayveDetectionEval(
        dc,
        'test/bevformer_small_wayve/Wed_Aug_17_09_27_00_2022/pts_bbox/results_wayve.json',
        'test/bevformer_small_wayve/Wed_Aug_17_09_27_00_2022/pts_bbox/labels_wayve.json',
        'test/bevformer_small_wayve/Tue_Aug_16_21_32_42_2022/pts_bbox',
    )
    de.main(render_curves=False)
