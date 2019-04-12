"""Converts YCB-Video data to SIXD format."""
import sys
import os
# Add parent directory to python path, to find libraries:
# sys.path.append('..') # Relative to CWD
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)))) # Relative to module, but cannot be used in notebooks

import argparse
from os.path import join
from shutil import copyfile as cp
import json
import yaml
import numpy
import scipy.io
from lib.rigidpose.sixd_toolkit.pysixd import inout


MM_SCALE = 1000

obj_labels = [
    '002_master_chef_can',
    '003_cracker_box',
    '004_sugar_box',
    '005_tomato_soup_can',
    '006_mustard_bottle',
    '007_tuna_fish_can',
    '008_pudding_box',
    '009_gelatin_box',
    '010_potted_meat_can',
    '011_banana',
    '019_pitcher_base',
    '021_bleach_cleanser',
    '024_bowl',
    '025_mug',
    '035_power_drill',
    '036_wood_block',
    '037_scissors',
    '040_large_marker',
    '051_large_clamp',
    '052_extra_large_clamp',
    '061_foam_brick',
]

def cls_idx2obj_label(cls_idx):
    return obj_labels[cls_idx-1]

def obj_label2cls_id(obj_label):
    return int(obj_label.split('_')[0])

def listdir(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


class Dataset():
    def __init__(self, args):
        self.source_path = args.source_path
        self.target_path = args.target_path
        self.subset_names = args.subset_names
        
    def get_sequences(self, subset):
        return sorted(listdir(join(self.source_path, subset)))


    # TODO
    # def create_cameras(self):
    #     cam_source = join(self.source_path, "cameras")
    #     for cam in listdir(cam_source):
    #         with open(join(cam_source, cam), 'r') as file:
    #             rig = json.read(file)["rig"]

    def process_ply(self):
        # Load & save ply files, scaling vertex coordinates
        # Copy texture maps, and reference from ply files
        path_models = join(self.source_path, 'models')
        os.makedirs(path_models, exist_ok=True)
        os.makedirs(join(self.target_path, 'models'), exist_ok=True)
        for model_dir in sorted(listdir(path_models)):
            model_id = obj_label2cls_id(model_dir)

            # In comments, since PLY loader / saver doesn't support this anyway
            # texture_in = join(self.source_path, 'models', model_dir, 'texture_map.png')
            # texture_out = join(self.target_path, 'models', 'texture_map_{:02d}.png'.format(model_id))
            # cp(texture_in, texture_out)

            ply_in = join(self.source_path, 'models', model_dir, 'textured.ply')
            ply_out = join(self.target_path, 'models', 'obj_{:02d}.ply'.format(model_id))

            model = inout.load_ply(ply_in)
            inout.save_ply(
                ply_out,
                model['pts'] * MM_SCALE,
                pts_colors = model['colors'],
                pts_normals = model['normals'],
                faces = model['faces'],
            )

    def create_models_info(self):
        model_dict = {}
        path_models = join(self.source_path, 'models')
        os.makedirs(path_models, exist_ok=True) 
        for model_dir in listdir(path_models):
            model_id = obj_label2cls_id(model_dir)
            with open(join(self.source_path, 'models', model_dir, 'points.xyz'), 'r') as file:
                points = MM_SCALE * numpy.loadtxt(file)
            min_xyz = points.min(axis=0).round(decimals=3).tolist()
            size_xyz = (points.max(axis=0) - points.min(axis=0)).round(decimals=3).tolist() 
            model_dict[model_id] = {'min_x': min_xyz[0],
                                    'min_y': min_xyz[1],
                                    'min_z': min_xyz[2],
                                    'size_x': size_xyz[0],
                                    'size_y': size_xyz[1],
                                    'size_z': size_xyz[2]}
        os.makedirs(join(self.target_path, 'models'), exist_ok=True)
        data = yaml.dump(model_dict)
        with open(join(self.target_path, 'models', 'models_info.yml'), 'w') as file:
            file.write(data)

    def create_gt_and_info(self, dir_src, dir_dst, filenames):
        gt_dict = {}
        info_dict = {}
        for file_idx, filename in enumerate(filenames):
            mat = scipy.io.loadmat(join(dir_src, filename + '-meta.mat'))
            with open(join(dir_src, filename + '-box.txt'), 'r') as file:
                bboxs = file.readlines()
            # gt.yml
            obj_list = []
            for i, bbox in enumerate(bboxs):
                pose = mat['poses'][..., i]
                obj = {'cam_R_m2c': pose[:, :3].flatten().tolist(),
                       'cam_t_m2c': (MM_SCALE * pose[:, 3]).tolist(),
                       'obj_bb': list(map(float, bbox.split()[1:])),
                       'obj_id': obj_label2cls_id(cls_idx2obj_label(int(mat['cls_indexes'][i])))}
                obj_list.append(obj)
            gt_dict[file_idx] = obj_list
            # info.yml
            info_dict[file_idx] = {'cam_K': mat['intrinsic_matrix'].flatten().tolist(),
                                   'depth_scale': int(mat['factor_depth'])}
        data = yaml.dump(gt_dict)
        with open(join(dir_dst, 'gt.yml'), 'w') as gt_file:
            gt_file.write(data)
        data = yaml.dump(info_dict)
        with open(join(dir_dst, 'info.yml'), 'w') as info_file:
            info_file.write(data)


    def copy_rgb(self, dir_src, dir_dst, filenames):
        dir_dst = join(dir_dst, 'rgb')
        os.makedirs(dir_dst, exist_ok=True)
        for file_idx, filename in enumerate(filenames):
            src = join(dir_src, filename + '-color.png')
            dst = join(dir_dst, "%06d" % file_idx + '.png')
            cp(src, dst)

    def copy_depth(self, dir_src, dir_dst, filenames):
        dir_dst = join(dir_dst, 'depth')
        os.makedirs(dir_dst, exist_ok=True)
        for file_idx, filename in enumerate(filenames):
            src = join(dir_src, filename + '-depth.png')
            dst = join(dir_dst, "%06d" % file_idx + '.png')
            cp(src, dst)


def parse_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Data converter',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source-path', required=True,
                        help='YCB-Video dataset root')
    parser.add_argument('--target-path', required=True,
                        help='Target Sixd dataset root')
    parser.add_argument('--subset-names', nargs='+', required=True,
                        help='names of subsets to convert')

    return parser.parse_args()


def main():
    ds = Dataset(parse_arguments())
    
    #ds.create_cameras()
    ds.create_models_info()
    ds.process_ply()
    for subset in ds.subset_names:
        seqs = ds.get_sequences(subset)
        for j, seq in enumerate(seqs):
            print("{} / {}...".format(j+1, len(seqs)))
            dir_src = join(ds.source_path, subset, seq)
            dir_dst = join(ds.target_path, subset, seq)
            os.makedirs(dir_dst, exist_ok=True)
            filenames = sorted(filename.split('-')[0] for filename in listdir(dir_src))
    
            ds.copy_depth(dir_src, dir_dst, filenames)
            ds.create_gt_and_info(dir_src, dir_dst, filenames)
            ds.copy_rgb(dir_src, dir_dst, filenames) 

if __name__ == '__main__':
    main()
