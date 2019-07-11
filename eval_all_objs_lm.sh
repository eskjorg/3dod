# exit when any command fails
set -e

OLD_EXPERIMENT_PREFIX=$1
NEW_EXPERIMENT_PREFIX=$2


# KP CLASSIF & VISIBILITY
# REPOPATH=/home/lucas/research/3dod
# DATAPATH=/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented3_format06
# CONTAINER=3dod-opengv
# CONFIGNAME=lm-kp-nonmutex

# KP-RCNN
REPOPATH=/home/lucas/research/3dod-kp-rcnn
DATAPATH=/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented3_format06
CONTAINER=3dod-kp-rcnn
CONFIGNAME=linemod-kp



TMP_SUFFIX=$(openssl rand -hex 4)
WS=/tmp/3dod-ws-$TMP_SUFFIX
rm -rf $WS
cp -r $REPOPATH $WS


OBJECTS=(duck)
# OBJECTS=(can)
# OBJECTS=(can duck)
# OBJECTS=(can cat)

# Discard driller (not present in validation sequence):
# OBJECTS=(duck can cat eggbox glue holepuncher ape)

# OBJECTS=(duck can cat driller eggbox glue holepuncher)
# OBJECTS=(duck can cat driller eggbox glue holepuncher ape)

for OBJ in ${OBJECTS[@]}; do
    echo "Removing experiment /hdd/lucas/out/3dod-experiments/$NEW_EXPERIMENT_PREFIX/$OBJ"
    rm -rf /hdd/lucas/out/3dod-experiments/$NEW_EXPERIMENT_PREFIX/$OBJ
done

for OBJ in ${OBJECTS[@]}; do
    ndocker \
        -e PYTHONPATH=/workspace/3dod \
        -w /workspace/3dod \
        -v $WS:/workspace/3dod \
        -v /hdd/lucas/out/3dod-experiments:/workspace/3dod/experiments \
        -v $DATAPATH:/datasets/occluded-linemod-augmented $CONTAINER python eval.py \
        --eval-mode val \
        --eval-mode train \
        --overwrite-experiment \
        --config-name $CONFIGNAME \
        --experiment-name $NEW_EXPERIMENT_PREFIX/$OBJ \
        --checkpoint-load-path /workspace/3dod/experiments/$OLD_EXPERIMENT_PREFIX/$OBJ/checkpoints/best_model.pth.tar \
        --group-labels $OBJ
done
rm -rf $WS
