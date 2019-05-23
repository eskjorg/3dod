"""Save results to disk."""
import os
import shutil
import json
import numpy as np
import torch

from collections import OrderedDict

from pyquaternion import Quaternion

import matplotlib
matplotlib.use('Agg')  # Overriding nuscenes backend
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from nuscenes.eval.eval_utils import category_to_detection_name
from nuscenes.eval.nuscenes_eval import NuScenesEval
from nuscenes.utils.data_classes import Box

from lib.utils import get_class_map
from lib.constants import NBR_KEYPOINTS, VISIB_TH


class KeypointEvaluator():

    def __init__(self, class_map, epoch_results, output_dir, mode):
        self._class_map = class_map
        self._epoch_results = epoch_results
        self._output_dir = output_dir
        self._mode = mode

    def _format_markdown_table(self, colnames, rows):
        rowdata = [colnames] + [['-']*len(colnames)] + rows
        return '\n'.join(['|'.join(map(str, row)) for row in rowdata])

    def _rowdict2rowdata(self, rowdict):
        """
        Convert a (possibly ordered) dict of rows (keys are row names), to a list of rows of the corresponding table, the first column being the header column.
        """
        table = [['**'+key+'**'] + row for key, row in rowdict.items()]
        return table[0], table[1:]

    def _coldict2rowdata(self, coldict):
        """
        Convert a (possibly ordered) dict of columns (keys are column names), to a list of rows of the corresponding table, the first row being the header row.
        """
        colnames, cols = zip(*coldict.items())
        rowdata = []
        rowdata += zip(*cols)
        return colnames, rowdata

    def run_eval_gridcell_stats(self, group_id):
        group_label = self._class_map.group_label_from_group_id(group_id)
    
        detection_stats = OrderedDict()
        for colname in ['kp_idx', 'avg_prec', 'avg_recall', '#avg_tp', '#avg_fp', '#avg_fn', '#avg_tn', '#gt_frames', 'freq of any res <= 5px']:
            detection_stats[colname] = [None]*NBR_KEYPOINTS
    
        for kp_idx in range(NBR_KEYPOINTS):
            detection_stats['kp_idx'][kp_idx] = kp_idx
            detection_stats['#avg_tp'][kp_idx] = np.mean([sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_tp'] for sample_result in self._epoch_results.values()])
            detection_stats['#avg_fp'][kp_idx] = np.mean([sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_fp'] for sample_result in self._epoch_results.values()])
            detection_stats['#avg_fn'][kp_idx] = np.mean([sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_fn'] for sample_result in self._epoch_results.values()])
            detection_stats['#avg_tn'][kp_idx] = np.mean([sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_tn'] for sample_result in self._epoch_results.values()])
            detection_stats['#gt_frames'][kp_idx] = np.mean([(sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_tp'] + sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_fn']) for sample_result in self._epoch_results.values()])
    
            detection_stats['avg_prec'][kp_idx] = detection_stats['#avg_tp'][kp_idx] / (detection_stats['#avg_tp'][kp_idx] + detection_stats['#avg_fp'][kp_idx])
            detection_stats['avg_recall'][kp_idx] = detection_stats['#avg_tp'][kp_idx] / (detection_stats['#avg_tp'][kp_idx] + detection_stats['#avg_fn'][kp_idx])
            # detection_stats['avg_prec'][kp_idx] = '{:0.2f} %'.format(100 * np.mean([sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_tp'] / (sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_tp'] + sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_fp']) for sample_result in self._epoch_results.values() if (sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_tp'] + sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_fn']) > 0]))
            # detection_stats['avg_recall'][kp_idx] = '{:0.2f} %'.format(100 * np.mean([sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_tp'] / (sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_tp'] + sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_fn']) for sample_result in self._epoch_results.values() if (sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_tp'] + sample_result[group_id]['kp_gridcell_stats'][kp_idx]['nbr_fn']) > 0]))
    
        for kp_idx in range(NBR_KEYPOINTS):
            detection_stats['#avg_tp'][kp_idx] = '{:0.2f}'.format(detection_stats['#avg_tp'][kp_idx])
            detection_stats['#avg_fp'][kp_idx] = '{:0.2f}'.format(detection_stats['#avg_fp'][kp_idx])
            detection_stats['#avg_fn'][kp_idx] = '{:0.2f}'.format(detection_stats['#avg_fn'][kp_idx])
            detection_stats['#avg_tn'][kp_idx] = '{:0.2f}'.format(detection_stats['#avg_tn'][kp_idx])
            detection_stats['#gt_frames'][kp_idx] = '{:0.2f}'.format(detection_stats['#gt_frames'][kp_idx])
            detection_stats['avg_prec'][kp_idx] = '{:0.2f} %'.format(100 * detection_stats['avg_prec'][kp_idx])
            detection_stats['avg_recall'][kp_idx] = '{:0.2f} %'.format(100 * detection_stats['avg_recall'][kp_idx])
    
        vis_path = os.path.join(self._output_dir, 'visual')
        # shutil.rmtree(vis_path, ignore_errors=True)
        writer = SummaryWriter(vis_path)
        writer.add_text(
            '{}_{}_{}'.format(self._mode, group_label, 'gridcell_detection_stats'),
            self._format_markdown_table(*self._coldict2rowdata(detection_stats)),
            0,
        )




    def run_eval_frame_stats(self, group_id):
        group_label = self._class_map.group_label_from_group_id(group_id)

        nbr_rows = NBR_KEYPOINTS + 1

        detection_stats = OrderedDict()
        colnames = [
            '#tp',
            '#fp',
            '#fn',
            '#tn',
            'median_lpeak_ratio',
            '#tp_acc (5px)',
            '#tp_inacc (5px)',
            '#tp_acc/(#tp+#fn) (5px)',
            '#tp_acc/(#tp+#fp+#tn+#fn) (5px)',
            '#tp_acc (10px)',
            '#tp_inacc (10px)',
            '#tp_acc/(#tp+#fn) (10px)',
            '#tp_acc/(#tp+#fp+#tn+#fn) (10px)',
        ]
        detection_stats['kp_idx'] = [None]*nbr_rows
        for key in colnames:
            detection_stats[key] = [None]*nbr_rows

        for kp_idx in range(NBR_KEYPOINTS):
            row_idx = kp_idx + 1

            detection_stats['kp_idx'][row_idx] = kp_idx
            detection_stats['#tp'][row_idx] = np.sum([sample_result[group_id]['kp_frame_stats'][kp_idx]['gt_gc_exist'] and sample_result[group_id]['kp_frame_stats'][kp_idx]['det_gc_exist'] for sample_result in self._epoch_results.values()], dtype=int)
            detection_stats['#fp'][row_idx] = np.sum([not sample_result[group_id]['kp_frame_stats'][kp_idx]['gt_gc_exist'] and sample_result[group_id]['kp_frame_stats'][kp_idx]['det_gc_exist'] for sample_result in self._epoch_results.values()], dtype=int)
            detection_stats['#tn'][row_idx] = np.sum([not sample_result[group_id]['kp_frame_stats'][kp_idx]['gt_gc_exist'] and not sample_result[group_id]['kp_frame_stats'][kp_idx]['det_gc_exist'] for sample_result in self._epoch_results.values()], dtype=int)
            detection_stats['#fn'][row_idx] = np.sum([sample_result[group_id]['kp_frame_stats'][kp_idx]['gt_gc_exist'] and not sample_result[group_id]['kp_frame_stats'][kp_idx]['det_gc_exist'] for sample_result in self._epoch_results.values()], dtype=int)
            detection_stats['#tp_acc (5px)'][row_idx] = np.sum([sample_result[group_id]['kp_frame_stats'][kp_idx]['tp_gc_exist'] and sample_result[group_id]['kp_frame_stats'][kp_idx]['min_resid_magnitude'] < 5.0 for sample_result in self._epoch_results.values()], dtype=int)
            detection_stats['#tp_acc (10px)'][row_idx] = np.sum([sample_result[group_id]['kp_frame_stats'][kp_idx]['tp_gc_exist'] and sample_result[group_id]['kp_frame_stats'][kp_idx]['min_resid_magnitude'] < 10.0 for sample_result in self._epoch_results.values()], dtype=int)
            # detection_stats['#tp_acc (10px)'][row_idx] = np.sum([sample_result[group_id]['kp_frame_stats'][kp_idx]['tp_acc'] for sample_result in self._epoch_results.values()], dtype=int)

            detection_stats['median_lpeak_ratio'][row_idx] = np.median([sample_result[group_id]['kp_frame_stats'][kp_idx]['lpeak_ratio'] for sample_result in self._epoch_results.values() if sample_result[group_id]['kp_frame_stats'][kp_idx]['tp_gc_exist']])

            detection_stats['#tp_inacc (5px)'][row_idx] = detection_stats['#tp'][row_idx] - detection_stats['#tp_acc (5px)'][row_idx]
            detection_stats['#tp_inacc (10px)'][row_idx] = detection_stats['#tp'][row_idx] - detection_stats['#tp_acc (10px)'][row_idx]
            detection_stats['#tp_acc/(#tp+#fn) (5px)'][row_idx] = detection_stats['#tp_acc (5px)'][row_idx] / (detection_stats['#tp'][row_idx] + detection_stats['#fn'][row_idx])
            detection_stats['#tp_acc/(#tp+#fn) (10px)'][row_idx] = detection_stats['#tp_acc (10px)'][row_idx] / (detection_stats['#tp'][row_idx] + detection_stats['#fn'][row_idx])
            detection_stats['#tp_acc/(#tp+#fp+#tn+#fn) (5px)'][row_idx] = detection_stats['#tp_acc (5px)'][row_idx] / (detection_stats['#tp'][row_idx] + detection_stats['#fp'][row_idx] + detection_stats['#tn'][row_idx] + detection_stats['#fn'][row_idx])
            detection_stats['#tp_acc/(#tp+#fp+#tn+#fn) (10px)'][row_idx] = detection_stats['#tp_acc (10px)'][row_idx] / (detection_stats['#tp'][row_idx] + detection_stats['#fp'][row_idx] + detection_stats['#tn'][row_idx] + detection_stats['#fn'][row_idx])

        # Averaging for top row
        detection_stats['kp_idx'][0] = 'avg'
        for key in colnames:
            detection_stats[key][0] = np.mean(detection_stats[key][1:])

        # ==========
        # FORMATTING
        # ==========
        for row_idx in range(nbr_rows):
            for key in ['kp_idx'] + colnames:
                if isinstance(detection_stats[key][row_idx], str):
                    continue
                if detection_stats[key][row_idx] is None:
                    detection_stats[key][row_idx] = ''
                elif '/' in key:
                    detection_stats[key][row_idx] = '{:0.2f} %'.format(100 * detection_stats[key][row_idx])
                elif key == 'median_lpeak_ratio':
                    detection_stats[key][row_idx] = '{:0.2f}'.format(detection_stats[key][row_idx])
                else:
                    detection_stats[key][row_idx] = '{}'.format(detection_stats[key][row_idx])
                    # detection_stats[key][row_idx] = '{:d}'.format(detection_stats[key][row_idx])

        vis_path = os.path.join(self._output_dir, 'visual')
        # shutil.rmtree(vis_path, ignore_errors=True)
        writer = SummaryWriter(vis_path)
        writer.add_text(
            '{}_{}_{}'.format(self._mode, group_label, 'frame_detection_stats'),
            self._format_markdown_table(*self._rowdict2rowdata(detection_stats)),
            # self._format_markdown_table(*self._coldict2rowdata(detection_stats)),
            0,
        )

        nbr_acc_kp_5px = []
        nbr_acc_kp_10px = []
        for sample_result in self._epoch_results.values():
            nbr_acc_kp_5px.append(np.sum([sample_result[group_id]['kp_frame_stats'][kp_idx]['tp_gc_exist'] and sample_result[group_id]['kp_frame_stats'][kp_idx]['min_resid_magnitude'] < 5.0 for kp_idx in range(NBR_KEYPOINTS)], dtype=int))
            nbr_acc_kp_10px.append(np.sum([sample_result[group_id]['kp_frame_stats'][kp_idx]['tp_gc_exist'] and sample_result[group_id]['kp_frame_stats'][kp_idx]['min_resid_magnitude'] < 10.0 for kp_idx in range(NBR_KEYPOINTS)], dtype=int))
        fig, axes_array = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=[17, 4],
            squeeze=False,
            tight_layout=True,
        )
        # fig, ax = plt.subplots()
        hist, _ = np.histogram(nbr_acc_kp_5px, bins=np.linspace(-0.5, NBR_KEYPOINTS+0.5, NBR_KEYPOINTS+2))
        axes_array[0,0].bar(list(range(NBR_KEYPOINTS+1)), hist, width=0.8, align='center')
        axes_array[0,0].set_xlabel('#KP < 5px somewhere')
        hist, _ = np.histogram(nbr_acc_kp_10px, bins=np.linspace(-0.5, NBR_KEYPOINTS+0.5, NBR_KEYPOINTS+2))
        axes_array[0,1].bar(list(range(NBR_KEYPOINTS+1)), hist, width=0.8, align='center')
        axes_array[0,1].set_xlabel('#KP < 10px somewhere')
        writer.add_figure('{}_{}_{}'.format(self._mode, group_label, 'nbr_acc_kp_hist'), fig, 0)

        # Unsure of the importance of calling close()... Might not be done in case of KeyboardInterrupt
        # https://stackoverflow.com/questions/44831317/tensorboard-unble-to-get-first-event-timestamp-for-run
        # https://stackoverflow.com/questions/33364340/how-to-avoid-suppressing-keyboardinterrupt-during-garbage-collection-in-python
        writer.close()


    def run_eval(self, group_id):
        self.run_eval_gridcell_stats(group_id)
        self.run_eval_frame_stats(group_id)
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
                # self.save_frame_kp_stats(
                #     frame_id,
                #     detections[frame_id], batch.annotation[frame_idx],
                #     cnn_outs['clsnonmutex'][0][frame_idx,:,:,:],
                #     batch.gt_map['clsnonmutex'][frame_idx,:,:,:],
                #     batch.gt_map['clsgroup'][frame_idx,:,:],
                # )
            # for frame_id, frame_detections in detections.items():
            #     self.save_frame_kp_stats(frame_id, frame_detections)

    def summarize_epoch(self, mode):
        if self._configs.data.dataformat == 'sixd_kp_instances':
            if not self._configs.logging.save_kp_stats:
                return None
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
                'kp_gridcell_stats': [],
                'kp_frame_stats': [],
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
                th_pred = 0.1
                th_gt = VISIB_TH
                pred_visib_binary = kp_data['pred_visib_map'] >= th_pred
                gt_visib_binary = kp_data['gt_visib_map'] >= th_gt
                sample_result[group_id]['kp_gridcell_stats'].append({
                    'nbr_tp': int(torch.sum(pred_visib_binary & gt_visib_binary).cpu().numpy()),
                    'nbr_fp': int(torch.sum(pred_visib_binary & (~gt_visib_binary)).cpu().numpy()),
                    'nbr_fn': int(torch.sum((~pred_visib_binary) & gt_visib_binary).cpu().numpy()),
                    'nbr_tn': int(torch.sum((~pred_visib_binary) & (~gt_visib_binary)).cpu().numpy()),
                })

                # Any positive grid cells at all for GT & pred respectively?
                gt_gc_exist = int(torch.sum(gt_visib_binary).cpu().numpy()) > 0
                det_gc_exist = int(torch.sum(pred_visib_binary).cpu().numpy()) > 0
                tp_gc_exist = int(torch.sum(pred_visib_binary & gt_visib_binary).cpu().numpy()) > 0
                if tp_gc_exist:
                    resid_map = kp_data['kp_map'] - torch.from_numpy(anno_lookup[class_id].keypoint.reshape((2,1,1))).float().cuda()
                    resid_at_tp_vec = torch.stack([
                        resid_map[0,:,:][pred_visib_binary & gt_visib_binary],
                        resid_map[1,:,:][pred_visib_binary & gt_visib_binary],
                    ])
                    resid_magnitude_at_tp_vec = torch.norm(resid_at_tp_vec, dim=0).cpu().numpy()
                    min_resid_magnitude = float(np.min(resid_magnitude_at_tp_vec))

                    # Determine ratio of estimated uncertainty between most certain & the best one
                    best_idx = np.argmin(resid_magnitude_at_tp_vec)
                    avg_ln_b_map = torch.mean(kp_data['kp_ln_b_map'], dim=0)
                    top_conf_ln_b = torch.min(avg_ln_b_map)
                    best_ln_b = avg_ln_b_map[pred_visib_binary & gt_visib_binary][best_idx]
                    lpeak_ratio = float(torch.exp(best_ln_b - top_conf_ln_b).cpu().numpy()) # Corresponds to ratio between likelihood peaks
                elif gt_gc_exist and det_gc_exist:
                    # Despite TP frame, there are no TP grid cells
                    min_resid_magnitude = None
                    lpeak_ratio = None
                else:
                    min_resid_magnitude = None
                    lpeak_ratio = None
                kp_frame_stats_data = {
                    'gt_gc_exist': gt_gc_exist,
                    'det_gc_exist': det_gc_exist,
                    'tp_gc_exist': tp_gc_exist,
                    'min_resid_magnitude': min_resid_magnitude,
                    'lpeak_ratio': lpeak_ratio,
                }
                sample_result[group_id]['kp_frame_stats'].append(kp_frame_stats_data)
                if 'visib_vec' not in kp_data:
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
