import sys
import os
# sys.path.append('../..')
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import numpy as np
import yaml
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.rigidpose.sixd_toolkit.pysixd import inout


# In[12]:


# SIXD_PATH = '/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented3_format06'
# obj_id = 9 # duck


# In[13]:


SIXD_PATH = '/home/lucas/datasets/pose-data/sixd/ycb-video2'
obj_id = 6 # mustard_bottle


# In[11]:


models_info = inout.load_yaml(os.path.join(SIXD_PATH, 'models', 'models_info.yml'))
models = {}
models[obj_id] = inout.load_ply(os.path.join(SIXD_PATH, 'models', 'obj_{:02}.ply'.format(obj_id)))
print("Obj {}: {} vertices, {} faces.".format(obj_id, len(models[obj_id]['pts']), len(models[obj_id]['faces'])))




N = 10

for j in range(N):
    nbr_pts = int(models[obj_id]['pts'].shape[0] * (j+1)/N)
    sample = np.random.choice(nbr_pts, size=10000)
    pts = models[obj_id]['pts'][sample,:]

    fig = plt.figure()
    ax = fig.add_subplot((111), projection='3d')
    ax.scatter(*pts.T)

plt.show()
