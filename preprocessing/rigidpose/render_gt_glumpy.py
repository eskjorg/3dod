
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
import png
from PIL import Image
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

# def read_png(filename, dtype=None, nbr_channels=3):
#     with open(filename, 'rb') as f:
#         data = png.Reader(f).read()[2]
#         if dtype is not None:
#             img = np.vstack(map(dtype, data))
#         else:
#             img = np.vstack(data)
#     shape = img.shape
#     assert shape[1] % nbr_channels == 0
#     img = np.reshape(img, (shape[0], shape[1]//nbr_channels, nbr_channels))
#     return img

def save_png(img, filename):
    shape = img.shape
    with open(filename, 'wb') as f:
        writer = png.Writer(
            width = shape[1],
            height = shape[0],
            bitdepth = 16,
            greyscale = False, # RGB
            alpha = False, # Not RGBA
        )
        writer.write(f, np.reshape(img, (shape[0], shape[1]*shape[2])))

models_info = read_yaml(os.path.join(SIXD_PATH, 'models', 'models_info.yml'))
models = {}
for j, obj_id in enumerate(sorted(models_info.keys())):
    print('Loading model {}/{}...'.format(j+1, len(models_info)))
    models[obj_id] = inout.load_ply(os.path.join(SIXD_PATH, 'models', 'obj_{:02}.ply'.format(obj_id)))

renderer = Renderer(
    [480, 640],
)
for obj_id, model in models.items():
    renderer._preprocess_object_model(obj_id, models[obj_id])

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

            if (j+1) % 100 == 0:
                print("subset {}, seq {}, frame {}/{}".format(subset, seq, j+1, len(fnames)))

            obj_id_list = []
            R_list = []
            t_list = []
            model_list = []
            for gt in gts[img_idx]:
                obj_id_list.append(gt['obj_id'])
                R_list.append(np.array(gt['cam_R_m2c']).reshape((3, 3)))
                t_list.append(np.array(gt['cam_t_m2c']).reshape((3,1)))
                model_list.append(models[gt['obj_id']])

            rgb, depth, seg, instance_seg, normal_map, corr_map = renderer.render(
                np.reshape(info['cam_K'], (3, 3)),
                R_list,
                t_list,
                obj_id_list,
                ambient_weight = 0.8,
                clip_near = 100, # mm
                clip_far = 10000, # mm
            )

            if not DRY_RUN:
                if render_seg:
                    seg_path = os.path.join(seg_dir, fname)
                    Image.fromarray(seg).save(seg_path)
                if render_instance_seg:
                    instance_seg_path = os.path.join(instance_seg_dir, fname)
                    Image.fromarray(instance_seg).save(instance_seg_path)
                if render_corr:
                    corr_path = os.path.join(corr_dir, fname)
                    save_png(corr_map, corr_path)
                if render_normals:
                    normals_path = os.path.join(normals_dir, fname)
                    save_png(normal_map, normals_path)
                if render_depth:
                    depth_rendered_path = os.path.join(depth_rendered_dir, fname)
                    Image.fromarray(depth).save(depth_path)
                if render_rgb:
                    rgb_rendered_path = os.path.join(rgb_rendered_dir, fname)
                    Image.fromarray(rgb).save(rgb_rendered_path)
            # assert False
