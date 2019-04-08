"""Save results to disk."""
import os
import shutil
import json
import numpy as np

from pyquaternion import Quaternion

import matplotlib
matplotlib.use('Agg')  # Overriding nuscenes backend
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from nuscenes.eval.eval_utils import category_to_detection_name
from nuscenes.eval.nuscenes_eval import NuScenesEval
from nuscenes.utils.data_classes import Box

from lib.utils import get_class_map
from lib.constants import NBR_KEYPOINTS


class KeypointEvaluator():

    def __init__(self, class_map, epoch_results, output_dir, mode):
        self._class_map = class_map
        self._epoch_results = epoch_results
        self._output_dir = output_dir
        self._mode = mode

    def run_eval(self, group_id):
        group_label = self._class_map.group_label_from_group_id(group_id)
        stats = {}
        for sample_token, sample_result in self._epoch_results.items():
            stats[sample_token] = {
                'kp_stats': {},
                'nbr_tp': 0,
                'nbr_fp': 0,
                'nbr_fn': 0,
                'nbr_tn': 0,
            }
            for kp_idx in range(NBR_KEYPOINTS):
                curr_kp_stats = {}

                # Tmp vars for convenience
                kp_gt_data = sample_result[group_id]['kp_gt'][kp_idx]
                kp_pred_data = sample_result[group_id]['kp_pred'][kp_idx]

                # Initialized to False
                curr_kp_stats['positive_anno'] = False
                curr_kp_stats['positive_pred'] = False

                # If most visible pixel is visible enough, annotation is considered visible:
                gt_visib_th = 0.5
                if kp_gt_data is not None and np.isfinite(kp_gt_data['max_visib']) and kp_gt_data['max_visib'] >= gt_visib_th:
                    curr_kp_stats['positive_anno'] = True

                # If predicted uncertainty is small enough, prediction is considered positive:
                if kp_pred_data is not None:
                    max_idx = np.argmin(np.array(kp_pred_data['kp_x_ln_b_vec']) + np.array(kp_pred_data['kp_y_ln_b_vec']))
                    geom_mean_std = np.sqrt(2.0 * np.exp(kp_pred_data['kp_x_ln_b_vec'][max_idx] + kp_pred_data['kp_y_ln_b_vec'][max_idx]))
                    pred_std_th = 5.0
                    # pred_std_th = 10.0
                    if geom_mean_std <= pred_std_th:
                        curr_kp_stats['positive_pred'] = True
                    curr_kp_stats['geom_mean_std'] = geom_mean_std
                    if kp_gt_data is not None:
                        res_x = kp_pred_data['kp_x_vec'][max_idx] - kp_gt_data['kp_x']
                        res_y = kp_pred_data['kp_y_vec'][max_idx] - kp_gt_data['kp_y']
                        curr_kp_stats['resid_magnitude'] = np.sqrt(res_x**2 + res_y**2)
                    else:
                        curr_kp_stats['resid_magnitude'] = None

                curr_kp_stats['tp'] = False
                curr_kp_stats['fp'] = False
                curr_kp_stats['fn'] = False
                curr_kp_stats['tn'] = False
                if curr_kp_stats['positive_pred'] and curr_kp_stats['positive_anno']:
                    curr_kp_stats['tp'] = True
                    stats[sample_token]['nbr_tp'] += 1
                elif curr_kp_stats['positive_pred'] and not curr_kp_stats['positive_anno']:
                    curr_kp_stats['fp'] = True
                    stats[sample_token]['nbr_fp'] += 1
                elif not curr_kp_stats['positive_pred'] and curr_kp_stats['positive_anno']:
                    curr_kp_stats['fn'] = True
                    stats[sample_token]['nbr_fn'] += 1
                elif not curr_kp_stats['positive_pred'] and not curr_kp_stats['positive_anno']:
                    curr_kp_stats['tn'] = True
                    stats[sample_token]['nbr_tn'] += 1

                stats[sample_token]['kp_stats'][kp_idx] = curr_kp_stats

        vis_path = os.path.join(self._output_dir, 'visual')
        shutil.rmtree(vis_path, ignore_errors=True)
        writer = SummaryWriter(vis_path)
        fig, axes_array = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=[10, 4],
            squeeze=False,
            tight_layout=True,
        )
        # fig, ax = plt.subplots()
        hist, _ = np.histogram([sample_stats['nbr_tp'] for sample_token, sample_stats in stats.items()], bins=np.linspace(-0.5, NBR_KEYPOINTS+0.5, NBR_KEYPOINTS+2))
        axes_array[0,0].bar(list(range(NBR_KEYPOINTS+1)), hist, width=0.8, align='center')
        axes_array[0,0].set_xlabel('#True Positives')
        # axes_array[0,0].set_title('True Positives')
        hist, _ = np.histogram([sample_stats['nbr_fp'] for sample_token, sample_stats in stats.items()], bins=np.linspace(-0.5, NBR_KEYPOINTS+0.5, NBR_KEYPOINTS+2))
        axes_array[0,1].bar(list(range(NBR_KEYPOINTS+1)), hist, width=0.8, align='center')
        axes_array[0,1].set_xlabel('#False Positives')
        # axes_array[0,1].set_title('False Positives')
        hist, _ = np.histogram([sample_stats['nbr_tn'] for sample_token, sample_stats in stats.items()], bins=np.linspace(-0.5, NBR_KEYPOINTS+0.5, NBR_KEYPOINTS+2))
        axes_array[1,0].bar(list(range(NBR_KEYPOINTS+1)), hist, width=0.8, align='center')
        axes_array[1,0].set_xlabel('#True Negatives')
        # axes_array[1,0].set_title('True Negatives')
        hist, _ = np.histogram([sample_stats['nbr_fn'] for sample_token, sample_stats in stats.items()], bins=np.linspace(-0.5, NBR_KEYPOINTS+0.5, NBR_KEYPOINTS+2))
        axes_array[1,1].bar(list(range(NBR_KEYPOINTS+1)), hist, width=0.8, align='center')
        axes_array[1,1].set_xlabel('#False Negatives')
        # axes_array[1,1].set_title('False Negatives')
        writer.add_figure('{}_{}_{}'.format(self._mode, group_label, 'conf_mat'), fig, 0)

        fig, axes_array = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=[10, 3],
            squeeze=False,
            tight_layout=True,
        )
        axes_array[0,0].hist([sample_stats['kp_stats'][kp_idx]['resid_magnitude'] for sample_token, sample_stats in stats.items() for kp_idx in range(NBR_KEYPOINTS) if sample_stats['kp_stats'][kp_idx]['tp']], bins=30)
        axes_array[0,0].set_xlabel('Residual Magnitude (px)')
        axes_array[0,0].set_title('True Positives')
        axes_array[0,1].hist([sample_stats['kp_stats'][kp_idx]['resid_magnitude'] for sample_token, sample_stats in stats.items() for kp_idx in range(NBR_KEYPOINTS) if sample_stats['kp_stats'][kp_idx]['fp'] and sample_stats['kp_stats'][kp_idx]['resid_magnitude'] is not None], bins=30)
        axes_array[0,1].set_xlabel('Residual Magnitude (px)')
        axes_array[0,1].set_title('False Positives')
        writer.add_figure('{}_{}_{}'.format(self._mode, group_label, 'resid_magnitude'), fig, 0)

        score = 9999999999999.0
        return score


class ResultSaver:
    """ResultSaver."""
    def __init__(self, configs):
        self._configs = configs
        self._class_map = get_class_map(self._configs)
        self._epoch_results = None

        self._result_dir = os.path.join(configs.experiment_path, 'detections')
        os.makedirs(self._result_dir, exist_ok=True)

        if configs.data.dataformat == 'nuscenes':
            self._nusc = configs.nusc
            self._init_results()
        elif configs.data.dataformat == 'sixd_kp_instances':
            self._epoch_results = {}

    def _init_results(self):
        self._epoch_results = {sample['token']: [] for sample in self._nusc.sample}

    def save(self, detections, mode, batch):
        if self._configs.logging.save_nuscenes_format:
            for data_token, frame_detections in detections.items():
                self.save_nuscenes_format(data_token, frame_detections)
        if self._configs.logging.save_kitti_format:
            for frame_id, frame_detections in detections.items():
                self.save_frame_kitti(frame_id, frame_detections, mode)
        if self._configs.logging.save_kp_stats:
            for frame_idx, frame_id in enumerate(batch.id):
                self.save_frame_kp_stats(frame_id, detections[frame_id], batch.annotation[frame_idx])
            # for frame_id, frame_detections in detections.items():
            #     self.save_frame_kp_stats(frame_id, frame_detections)

    def summarize_epoch(self, mode):
        if self._configs.data.dataformat == 'sixd_kp_instances':
            scores = {}
            for group_id in self._class_map.get_group_ids():
                group_label = self._class_map.group_label_from_group_id(group_id)
                output_dir = os.path.join(self._result_dir, 'kp_eval', group_label)
                kp_eval = KeypointEvaluator(self._class_map, self._epoch_results, output_dir, mode)
                score = kp_eval.run_eval(group_id)
                scores[group_id] = score
            result_path = self.write_to_file()
            return np.mean(list(scores.values()))
        elif self._configs.data.dataformat != 'nuscenes':
            return None  # TODO: implement evaluation for other datasets
        result_path = self.write_to_file()
        eval_set = 'teaser_' + mode
        if mode == 'test':  # TODO: until full dataset released
            eval_set = 'teaser_val'
        output_dir = os.path.join(self._result_dir, 'nuscenes_eval')
        nusc_eval = NuScenesEval(nusc=self._nusc,
                                 result_path=result_path,
                                 eval_set=eval_set,
                                 output_dir=output_dir)
        all_metrics = nusc_eval.run_eval()
        score = all_metrics[self._configs.evaluation.score]
        return score

    def write_to_file(self):
        if self._configs.data.dataformat == 'nuscenes':
            result_path = os.path.join(self._result_dir, 'nuscenes_results.json')
        elif self._configs.data.dataformat == 'sixd_kp_instances':
            result_path = os.path.join(self._result_dir, 'kp_results.json')
        with open(result_path, 'w') as file:
            json.dump(self._epoch_results, file, indent=4)
        if self._configs.data.dataformat == 'nuscenes':
            self._init_results()
        elif self._configs.data.dataformat == 'sixd_kp_instances':
            self._epoch_results = {}
        return result_path

    def save_frame_kp_stats(self, sample_token, frame_detections, frame_annotations):
        sample_result = {}
        anno_lookup = dict(zip([anno.cls for anno in frame_annotations], frame_annotations))
        for group_id in frame_detections:
            sample_result[group_id] = {
                'kp_gt': [],
                'kp_pred': [],
            }
            for kp_idx in range(NBR_KEYPOINTS):
                class_id = self._class_map.class_id_from_group_id_and_kp_idx(group_id, kp_idx)

                if class_id in anno_lookup:
                    max_visib = np.max(anno_lookup[class_id].keypoint_detectability.numpy())
                    sample_result[group_id]['kp_gt'].append({
                        'kp_x': float(anno_lookup[class_id].keypoint[0]),
                        'kp_y': float(anno_lookup[class_id].keypoint[1]),
                        'max_visib': float(max_visib),
                    })
                else:
                    sample_result[group_id]['kp_gt'].append(None)

                kp_data = frame_detections[group_id]['keypoints'][kp_idx]
                if kp_data is None:
                    sample_result[group_id]['kp_pred'].append(None)
                else:
                    sample_result[group_id]['kp_pred'].append({
                        'visib_vec': kp_data['visib_vec'].cpu().numpy().tolist(),
                        'idx_x_vec': kp_data['idx_x_vec'].cpu().numpy().tolist(),
                        'idx_y_vec': kp_data['idx_y_vec'].cpu().numpy().tolist(),
                        'kp_x_vec': kp_data['kp_x_vec'].cpu().numpy().tolist(),
                        'kp_y_vec': kp_data['kp_y_vec'].cpu().numpy().tolist(),
                        'kp_x_ln_b_vec': kp_data['kp_x_ln_b_vec'].cpu().numpy().tolist(),
                        'kp_y_ln_b_vec': kp_data['kp_y_ln_b_vec'].cpu().numpy().tolist(),
                    })
        self._epoch_results[sample_token] = sample_result

    def save_nuscenes_format(self, data_token, frame_detections):
        # TODO: Make sure that not multiple sample_datas write to the same sample.
        sample_results = []
        sample_data = self._nusc.get('sample_data', data_token)
        sample_token = sample_data['sample_token']

        for detection in frame_detections:
            if min(detection['size']) < 0:
                print('Negative size. Applying `abs()`')
            location, rotation = self.get_nusc_global_pose(detection, sample_data)
            sample_result = {
                "sample_token": sample_token,
                "translation": location.tolist(),
                "size": abs(detection['size']).tolist(),
                "rotation": rotation.normalised.elements.tolist(),
                "velocity": 3 * [float('nan')],
                "detection_name": category_to_detection_name(detection['cls']),
                "detection_score": float(detection['confidence']),
                "attribute_scores": 8 * [-1]
            }
            sample_results.append(sample_result)
        self._epoch_results[sample_token] = sample_results

    def get_nusc_global_pose(self, box, sample_data):
        box = Box(center=box['location'],
                  size=box['size'][[1, 2, 0]],  # HWL to WLH
                  orientation=Quaternion(axis=(0, 0, 1), angle=box['rotation_y']))

        pose_record = self._nusc.get('ego_pose', sample_data['ego_pose_token'])
        cs_record = self._nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

        # Move box to ego vehicle coord system.
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))

        # Move box to global coord system.
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))

        return box.center, box.orientation

    def save_frame_kitti(self, frame_id, frame_detections, mode):
        save_dir = os.path.join(self._result_dir, mode, 'kitti_format')
        os.makedirs(save_dir, exist_ok=True)

        lines_to_write = []
        for detection in frame_detections:
            bbox2d = self._clip_bbox(detection.bbox2d)
            size = detection.get('size', [-1] * 3)
            location = detection.get('location', [-1000] * 3)
            write_line = ('{cls:} -1 -1 {alpha:=.5g} '
                          '{left:=.2f} {top:=.2f} {right:=.2f} {bottom:=.2f} '
                          '{height:=.3g} {width:=.3g} {length:=.3g} '
                          '{x:=.4g} {y:=.4g} {z:=.4g} {rotation_y:=.3g} '
                          '{score:=.6f}\n'.
                          format(cls=detection.cls,
                                 alpha=detection.get('alpha', -10),
                                 left=bbox2d[0],
                                 top=bbox2d[1],
                                 right=bbox2d[2],
                                 bottom=bbox2d[3],
                                 height=size[0],
                                 width=size[1],
                                 length=size[2],
                                 x=location[0],
                                 y=location[1] + size[0] / 2,  # Kitti format
                                 z=location[2],
                                 rotation_y=detection.get('rotation_y', -10),
                                 score=detection.confidence))
            if (np.array(size) > np.array([0.5, 0.2, 0.1])).all():
                lines_to_write.append(write_line)
            else:
                print('Warning, negative size: Skipping writing')
        with open(os.path.join(save_dir, '%6d.txt' % int(frame_id)), 'w') as file:
            file.writelines(lines_to_write)

    def _clip_bbox(self, bbox):
        return [max(0, bbox[0]),
                max(0, bbox[1]),
                min(self._configs.data.img_dims[1] - 1, bbox[2]),
                min(self._configs.data.img_dims[0] - 1, bbox[3])]
