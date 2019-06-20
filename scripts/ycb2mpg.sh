OUT_DIR=/home/lucas/datasets/ycb-video/mpg
mkdir -p $OUT_DIR
for SEQ in /home/lucas/datasets/ycb-video/YCB_Video_Dataset/data/*; do
    SEQ_IDX=$(basename $SEQ)
    ffmpeg -framerate 100 -i $SEQ/%06d-color.png $OUT_DIR/$SEQ_IDX.mpg || exit 1
done
