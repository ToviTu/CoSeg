#!/bin/bash

# DIR="/scratch/t.tovi/datasets/"
# DIR="/home/tovi/Downloads/
DIR="../../datasets"

# Download the datasets
# wget --directory-prefix=$DIR http://images.cocodataset.org/zips/train2017.zip
# wget --directory-prefix=$DIR http://images.cocodataset.org/zips/val2017.zip
# wget --directory-prefix=$DIR http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

# change to wget if not on mac
# curl -o "$DIR/VOCtrainval_11-May-2012.tar" "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

# Cityscapes datasets
# cityscapes login info
# ask Ben for login info
    # might be hard to download through curl bc of login, so if we want this one need to download locally and transfer to server

# ADE20K dataset
# ask Ben for login info

# Pascal Context (2010) dataset
curl -o "$DIR/VOCtrainval_03-May-2010.tar" "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"

# Unpack the datasets
# mkdir -p ${DIR}/COCO_stuff_images
# mkdir -p ${DIR}/COCO_stuff_annotations
mkdir -p ${DIR}/VOC2012
mkdir -p ${DIR}/Cityscapes
mkdir -p ${DIR}/ADE20K
mkdir -p ${DIR}/VOC2010PascalContext

# unzip ${DIR}/train2017.zip -d ${DIR}/COCO_stuff_images/
# unzip ${DIR}/val2017.zip -d ${DIR}/COCO_stuff_images/
# unzip ${DIR}/stuffthingmaps_trainval2017.zip -d ${DIR}/COCO_stuff_annotations/
unzip ${DIR}/VOCtrainval_11-May-2012.tar -d ${DIR}/VOC2012/
unzip ${DIR}/Cityscapes_gtFine.zip -d ${DIR}/Cityscapes/

unzip ${DIR}/VOCtrainval_03-May-2010.tar -d ${DIR}/VOC2010PascalContext/
