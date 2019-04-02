FROM nvcr.io/nvidia/pytorch:19.03-py3
# NOTE: 19.03 still use old APEX (with old API)

RUN conda update -n base -c conda-forge conda
RUN conda install -y -c conda-forge \
    attrdict \
    opencv \
    tensorboardX
RUN pip install nuscenes-devkit
RUN pip install --upgrade numpy  # Currenty needed to cope with old numpy in nuScenes
RUN pip install git+https://github.com/facebookresearch/maskrcnn-benchmark.git@master