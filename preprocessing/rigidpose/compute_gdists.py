import sys
import os
sys.path.append('../..')
#sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

import yaml
import gdist
from lib.rigidpose.sixd_toolkit.pysixd import inout
import numpy as np

SIXD_PATH = '/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented2cc_gdists'

# Load models
models_info = inout.load_yaml(os.path.join(SIXD_PATH, 'models', 'models_info.yml'))
models = {}
for obj_id in models_info:
    models[obj_id] = inout.load_ply(os.path.join(SIXD_PATH, 'models', 'obj_{:02}.ply'.format(obj_id)))
    print("Obj {}: {} vertices, {} faces.".format(obj_id, len(models[obj_id]['pts']), len(models[obj_id]['faces'])))


def find_closest_vtx(x, y, z, vertices):
    assert vertices.shape[1] == 3
    distances = np.linalg.norm(vertices - np.array([[x, y, z]]), axis=1)
    vtx_idx = np.argmin(distances)
    return vtx_idx

def compute_gdists_on_models(models, models_info):
    gdists = {}
    obj_cnt = 0
    for obj_id, model in models.items():
        obj_cnt += 1
        nbr_vtx = model['pts'].shape[0]
        nbr_kp = len(models_info[obj_id]['kp_x'])
        gdists[obj_id] = {}
        for kp_idx, kp_coords in enumerate(zip(models_info[obj_id]['kp_x'], models_info[obj_id]['kp_y'], models_info[obj_id]['kp_z'])):
            kp_vtx_idx = find_closest_vtx(*kp_coords, model['pts'])
            print("Obj {}/{}: {}, keypoint {}/{}".format(obj_cnt, len(models), obj_id, kp_idx+1, nbr_kp))
            gdists[obj_id][kp_idx] = gdist.compute_gdist(
                model['pts'].astype(np.float64),
                model['faces'].astype(np.int32),
                source_indices = np.array([kp_vtx_idx], np.int32),
                #target_indices = np.array(list(range(nbr_vtx)), np.int32),
                #max_distance = 100.0,
            )
    #        colors = gdist_to_kp_per_vtx[:,np.newaxis]
    #        colors = 255.999*(1.0-colors/np.max(colors))
    #        models[obj_id]['colors'][:,:] = colors.astype('uint8')
    #        inout.save_ply(
    #            '/tmp/test.ply',
    #            models[obj_id]['pts'],
    #            pts_colors = models[obj_id]['colors'],
    #            pts_normals = models[obj_id]['normals'],
    #            faces = models[obj_id]['faces'],
    #        )
    #        break
    #    break
    return gdists
gdists = compute_gdists_on_models(models, models_info)


# Store gdists as yaml
with open(os.path.join(SIXD_PATH, 'models', 'gdists.yml'), 'w') as f:
    yaml.dump(gdists, f, Dumper=yaml.CDumper)
