# exit when any command fails
set -e

EXPERIMENT_PREFIX=$1

TMP_SUFFIX=$(openssl rand -hex 4)
WS=/tmp/3dod-ws-$TMP_SUFFIX
rm -rf $WS
cp -r /home/lucas/research/3dod $WS

# Discard driller (not present in validation sequence):
OBJECTS=(duck can cat eggbox glue holepuncher ape)
# OBJECTS=(duck can cat driller eggbox glue holepuncher ape)

for OBJ in ${OBJECTS[@]}; do
    echo "Removing experiment /hdd/lucas/out/3dod-experiments/$EXPERIMENT_PREFIX/$OBJ"
    rm -rf /hdd/lucas/out/3dod-experiments/$EXPERIMENT_PREFIX/$OBJ
done

for OBJ in ${OBJECTS[@]}; do
    ndocker \
        -e PYTHONPATH=/workspace/3dod \
        -w /workspace/3dod \
        -v $WS:/workspace/3dod \
        -v /hdd/lucas/out/3dod-experiments:/workspace/3dod/experiments \
        -v /home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented3_format06:/datasets/occluded-linemod-augmented 3dod-opengv python train.py \
        --overwrite-experiment \
        --config-name lm-kp-nonmutex \
        --experiment-name $EXPERIMENT_PREFIX/$OBJ \
        --train-seqs train_unoccl/* \
        --group-labels $OBJ
done
rm -rf $WS
