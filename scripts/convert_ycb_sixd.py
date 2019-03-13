"""Converts YCB-Video data to SIXD format."""
import argparse
import os
from os.path import join
from shutil import copyfile as cp
import json
import yaml
import numpy
import scipy.io


MM_SCALE = 1000


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
        return listdir(join(self.source_path, subset))


    # TODO
    # def create_cameras(self):
    #     cam_source = join(self.source_path, "cameras")
    #     for cam in listdir(cam_source):
    #         with open(join(cam_source, cam), 'r') as file:
    #             rig = json.read(file)["rig"]

    # TODO
    # def create_ply():
    #    pass

    def create_models_info(self):
        model_dict = {}
        path_models = join(self.source_path, 'models')
        os.makedirs(path_models, exist_ok=True) 
        for model_dir in listdir(path_models):
            model_id = int(model_dir.split('_')[0])
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
        with open(join(self.target_path, 'models', 'models_info.yaml'), 'w') as file:
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
                       'obj_id': int(mat['cls_indexes'][i])}
                obj_list.append(obj)
            gt_dict[file_idx] = obj_list
            # info.yml
            info_dict[file_idx] = {'cam_K': [mat['intrinsic_matrix'].flatten().tolist()],
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
    #ds.create_ply()
    for subset in ds.subset_names:
        for seq in ds.get_sequences(subset):
            dir_src = join(ds.source_path, subset, seq)
            dir_dst = join(ds.target_path, subset, seq)
            os.makedirs(dir_dst, exist_ok=True)
            filenames = sorted(filename.split('-')[0] for filename in listdir(dir_src))
            
            ds.copy_depth(dir_src, dir_dst, filenames)
            ds.create_gt_and_info(dir_src, dir_dst, filenames)
            ds.copy_rgb(dir_src, dir_dst, filenames) 

if __name__ == '__main__':
    main()
