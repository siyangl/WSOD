DATASET=$1
IMAGE_SET=$2
OUTPUT_DIR=$3
NETWORK=vgg_fcn
IMAGE_SIZE=321
STEPS=10000
python train/train_classification.py \
--max_steps=${STEPS} \
--batch_size=16 \
--pretrained_model=./data/imagenet_models/SEC_init.ckpt \
--dataset_name=${DATASET} \
--image_set=${IMAGE_SET} \
--train_dir=${OUTPUT_DIR} \
--image_size=${IMAGE_SIZE} \
--flip_images=True \
--crop_images=True \
--multilabel=True \
--network=${NETWORK} \
--dropout=True \

python eval/eval_classification.py \
--dataset_name=${DATASET} \
--checkpoint=${OUTPUT_DIR}/model.ckpt-${STEPS} \
--multilabel=True \
--network=${NETWORK} \
--image_size=${IMAGE_SIZE} \
