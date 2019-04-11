"""
Render ground truth instance segmentations, correspondence maps and normal maps from pose annotations.
"""

import sys
import os
# Add parent directory to python path, to find libraries:
# sys.path.append('../..') # Relative to CWD
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) # Relative to module, but cannot be used in notebooks

from lib.rigidpose.sixd_toolkit.pysixd import inout
from preprocessing.rigidpose.glumpy_renderer import Renderer
from lib.utils import listdir_nohidden
import yaml
import numpy as np
import shutil


DRY_RUN = False
LINEMOD_FLAG = False

if LINEMOD_FLAG:
    SIXD_PATH = '/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented2bb_instance_idx_fix'
    SUBSETS = [subset for subset in listdir_nohidden(SIXD_PATH) if subset.startswith('train') or subset.startswith('test')]
else:
    SIXD_PATH = '/home/lucas/datasets/pose-data/sixd/ycb-video2'
    SUBSETS = ['data']


def read_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.CLoader)

models_info = read_yaml(os.path.join(SIXD_PATH, 'models', 'models_info.yml'))
models = {}
for obj_id in models_info:
    models[obj_id] = inout.load_ply(os.path.join(SIXD_PATH, 'models', 'obj_{:02}.ply'.format(obj_id)))

for subset in SUBSETS:
    for seq in sorted(listdir_nohidden(os.path.join(SIXD_PATH, subset))):
        render_seg = False if LINEMOD_FLAG and subset == 'train_aug' else True
        render_instance_seg = False if LINEMOD_FLAG and subset == 'train_aug' else True
        render_corr = True
        render_normals = True if LINEMOD_FLAG else False
        render_depth = True if LINEMOD_FLAG else False
        render_rgb = True if LINEMOD_FLAG else False

        rgb_dir = os.path.join(SIXD_PATH, subset, seq, 'rgb')
        seg_dir = os.path.join(SIXD_PATH, subset, seq, 'seg')
        instance_seg_dir = os.path.join(SIXD_PATH, subset, seq, 'instance_seg')
        corr_dir = os.path.join(SIXD_PATH, subset, seq, 'obj')
        normals_dir = os.path.join(SIXD_PATH, subset, seq, 'normals')
        depth_rendered_dir = os.path.join(SIXD_PATH, subset, seq, 'depth_rendered')
        rgb_rendered_dir = os.path.join(SIXD_PATH, subset, seq, 'rgb_rendered')

        if not DRY_RUN:
            if render_seg:
                if os.path.exists(seg_dir):
                    shutil.rmtree(seg_dir)
                os.makedirs(seg_dir)
            if render_instance_seg:
                if os.path.exists(instance_seg_dir):
                    shutil.rmtree(instance_seg_dir)
                os.makedirs(instance_seg_dir)
            if render_corr:
                if os.path.exists(corr_dir):
                    shutil.rmtree(corr_dir)
                os.makedirs(corr_dir)
            if render_normals:
                if os.path.exists(normals_dir):
                    shutil.rmtree(normals_dir)
                os.makedirs(normals_dir)
            if render_depth:
                if os.path.exists(depth_rendered_dir):
                    shutil.rmtree(depth_rendered_dir)
                os.makedirs(depth_rendered_dir)
            if render_rgb:
                if os.path.exists(rgb_rendered_dir):
                    shutil.rmtree(rgb_rendered_dir)
                os.makedirs(rgb_rendered_dir)

        gts = read_yaml(os.path.join(SIXD_PATH, subset, seq, 'gt.yml'))
        infos = read_yaml(os.path.join(SIXD_PATH, subset, seq, 'info.yml'))

        fnames = list(sorted(listdir_nohidden(rgb_dir)))
        for j, fname in enumerate(fnames):
            img_idx = int(fname.split('.')[0])

            info = infos[img_idx]

            if (j+1) % 1 == 0:
                print("subset {}, seq {}, frame {}/{}".format(subset, seq, j+1, len(fnames)))

            seg_path = os.path.join(seg_dir, fname) if render_seg else None
            instance_seg_path = os.path.join(instance_seg_dir, fname) if render_instance_seg else None
            corr_path = os.path.join(corr_dir, fname) if render_corr else None
            normals_path = os.path.join(normals_dir, fname) if render_normals else None
            depth_rendered_path = os.path.join(depth_rendered_dir, fname) if render_depth else None
            rgb_rendered_path = os.path.join(rgb_rendered_dir, fname) if render_rgb else None

            obj_id_list = []
            R_list = []
            t_list = []
            model_list = []
            for gt in gts[img_idx]:
                obj_id_list.append(gt['obj_id'])
                R_list.append(np.array(gt['cam_R_m2c']).reshape((3, 3)))
                t_list.append(np.array(gt['cam_t_m2c']).reshape((3,1)))
                model_list.append(models[gt['obj_id']])

            renderer = Renderer(
                [480, 640],
                np.reshape(info['cam_K'], (3, 3)),
                clip_near = 100, # mm
                clip_far = 10000, # mm
            )

            rgb, depth, seg, instance_seg, normal_map, corr_map = renderer.render(
                model_list,
                R_list,
                t_list,
                obj_id_list,
                texture_map_list = None,
                surf_color_list = None,
                ambient_weight = 0.8,
            )

            # if not DRY_RUN:
            #     exit_code = os.system(cmd)
            #     if exit_code != 0:
            #         print("Fail!")
            #         sys.exit(exit_code)
