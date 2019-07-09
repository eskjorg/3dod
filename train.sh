# exit when any command fails
set -e

EXPERIMENT_PREFIX=$1
rm -rf /hdd/lucas/out/3dod-experiments/$EXPERIMENT_PREFIX

ndocker \
    -e PYTHONPATH=/workspace/3dod \
    -w /workspace/3dod \
    -v $PWD:/workspace/3dod \
    -v /hdd/lucas/out/3dod-experiments:/workspace/3dod/experiments \
    -v /home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented2_gdists:/datasets/occluded-linemod-augmented \
    3dod-kp-rcnn python train.py \
    --config-name linemod-kp \
    --overwrite-experiment \
    --experiment-name $EXPERIMENT_PREFIX
