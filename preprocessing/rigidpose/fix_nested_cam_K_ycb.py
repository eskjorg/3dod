import os
import yaml
import glob

root_path = '/home/lucas/datasets/pose-data/sixd/ycb-video2'

seq_paths = sorted(map(lambda gt_path: os.path.dirname(gt_path), glob.glob(os.path.join(root_path, '*', '*', 'gt.yml'))))

for seq_path in seq_paths:
    print(seq_path)

    with open(os.path.join(seq_path, 'info.yml'), 'r') as f:
        info_yaml = yaml.load(f, Loader=yaml.CLoader)

    for frame_idx, frame_info in info_yaml.items():
        frame_info['cam_K'] = frame_info['cam_K'][0]

    with open(os.path.join(seq_path, 'info.yml'), 'w') as f:
        yaml.dump(info_yaml, f, Dumper=yaml.CDumper)
