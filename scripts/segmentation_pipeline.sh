DATASET=$1
IMAGE_SET=$2
OUTPUT_DIR=$3
STEP=8000
python train/train_segmentation.py \
--max_steps=${STEP} \
--batch_size=16 \
--pretrained_model=./data/imagenet_models/SEC_init.ckpt \
--dataset_name=${DATASET} \
--image_set=${IMAGE_SET} \
--train_dir=${OUTPUT_DIR} \
--data_dir=./data/seg/VOC2007 \
--dropout=True

python eval/inference_segmentation.py \
--checkpoint=${OUTPUT_DIR}/model.ckpt-${STEP} \
--image_dir=../Dataset/VOCdevkit/VOC2007/JPEGImages \
--image_list_file=../Dataset/VOCdevkit/VOC2007/ImageSets/Main/${IMAGE_SET}.txt \
--annotation_dir=../Dataset/VOCdevkit/VOC2007/Annotations_bak_true_anno

python eval/eval_segmentation.py \
--result_dir=${OUTPUT_DIR}/label \
--annotation_dir=../Dataset/VOCdevkit/VOC2007/Annotations_bak_true_anno \
--image_list=../Dataset/VOCdevkit/VOC2007/ImageSets/Main/${IMAGE_SET}.txt
