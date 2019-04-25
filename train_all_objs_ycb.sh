# exit when any command fails
set -e

EXPERIMENT_PREFIX=$1

TMP_SUFFIX=$(openssl rand -hex 4)
WS=/tmp/3dod-ws-$TMP_SUFFIX
rm -rf $WS
cp -r /home/lucas/research/3dod $WS
OBJECTS=(master_chef_can cracker_box sugar_box tomato_soup_can mustard_bottle tuna_fish_can pudding_box gelatin_box potted_meat_can banana pitcher_base bleach_cleanser bowl mug power_drill wood_block scissors large_marker large_clamp extra_large_clamp foam_brick)

for OBJ in ${OBJECTS[@]}; do
    echo "Removing experiment /hdd/lucas/out/3dod-experiments/$EXPERIMENT_PREFIX-$OBJ"
    rm -rf /hdd/lucas/out/3dod-experiments/$EXPERIMENT_PREFIX-$OBJ
done

for OBJ in ${OBJECTS[@]}; do
    ndocker \
        -e PYTHONPATH=/workspace/3dod \
        -w /workspace/3dod \
        -v $WS:/workspace/3dod \
        -v /hdd/lucas/out/3dod-experiments:/workspace/3dod/experiments \
        -v /home/lucas/datasets/pose-data/sixd/ycb-video2:/datasets/ycb-video 3dod-opengv python train.py \
        --overwrite-experiment \
        --config-name ycb-kp-nonmutex \
        --experiment-name $EXPERIMENT_PREFIX-$OBJ \
        --train-seqs train/* \
        --group-labels $OBJ
done
rm -rf $WS
