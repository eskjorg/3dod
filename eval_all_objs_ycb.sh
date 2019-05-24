# exit when any command fails
set -e

OLD_EXPERIMENT_PREFIX=$1
NEW_EXPERIMENT_PREFIX=$2

TMP_SUFFIX=$(openssl rand -hex 4)
WS=/tmp/3dod-ws-$TMP_SUFFIX
rm -rf $WS
cp -r /home/lucas/research/3dod $WS

# OBJECTS=(master_chef_can)
# OBJECTS=(cracker_box)
# OBJECTS=(master_chef_can cracker_box)
# OBJECTS=(sugar_box tomato_soup_can tuna_fish_can pudding_box gelatin_box)
OBJECTS=(master_chef_can cracker_box sugar_box tomato_soup_can mustard_bottle tuna_fish_can pudding_box gelatin_box potted_meat_can banana pitcher_base bleach_cleanser bowl mug power_drill wood_block scissors large_marker large_clamp extra_large_clamp foam_brick)

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
        -v /home/lucas/datasets/pose-data/sixd/ycb-video2:/datasets/ycb-video 3dod-opengv python eval.py \
        --eval-mode val \
        --eval-mode train \
        --overwrite-experiment \
        --config-name ycb-kp-nonmutex \
        --experiment-name $NEW_EXPERIMENT_PREFIX/$OBJ \
        --checkpoint-load-path /workspace/3dod/experiments/$OLD_EXPERIMENT_PREFIX/$OBJ/checkpoints/best_model.pth.tar \
        --group-labels $OBJ
        # --train-seqs val/0019 \
        # --train-seqs train/0078 \
done
rm -rf $WS
