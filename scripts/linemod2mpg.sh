OUT_DIR=/home/lucas/datasets/pose-data/linemod-mpg
mkdir -p $OUT_DIR
for SEQ in /home/lucas/datasets/pose-data/linemod/original/data/*; do
    SEQ_NAME=$(basename $SEQ)
    ffmpeg -r 4 -i $SEQ/data/color%d.jpg -r 25 $OUT_DIR/$SEQ_NAME.mpg || exit 1
done
