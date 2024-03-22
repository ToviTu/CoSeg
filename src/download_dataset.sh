#!/bin/bash

# DIR="/scratch/t.tovi/datasets/"
# DIR="/home/tovi/Downloads/
DIR="../../datasets"

# Download the datasets
# wget --directory-prefix=$DIR http://images.cocodataset.org/zips/train2017.zip
# wget --directory-prefix=$DIR http://images.cocodataset.org/zips/val2017.zip
# wget --directory-prefix=$DIR http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
# curl -o "$DIR/VOCtrainval_11-May-2012.tar" "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

# Unpack the datasets
# mkdir -p ${DIR}/COCO_stuff_images
# mkdir -p ${DIR}/COCO_stuff_annotations
mkdir -p ${DIR}/VOC2012
# unzip ${DIR}/train2017.zip -d ${DIR}/COCO_stuff_images/
# unzip ${DIR}/val2017.zip -d ${DIR}/COCO_stuff_images/
# unzip ${DIR}/stuffthingmaps_trainval2017.zip -d ${DIR}/COCO_stuff_annotations/
unzip ${DIR}/VOCtrainval_11-May-2012.tar -d ${DIR}/VOC2012/