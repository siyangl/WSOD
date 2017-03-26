DATASET=$1
IMAGE_SET=$2
OUTPUT_DIR=$3
python train/train_detection_mil.py \
--max_steps=10000 \
--batch_size=4 \
--pretrained_model=./data/imagenet_models/VGG16.ckpt \
--dataset_name=${DATASET} \
--image_set=${IMAGE_SET} \
--train_dir=${OUTPUT_DIR} \
--flip_images=False \
--crop_images=False \
--dropout=True \
