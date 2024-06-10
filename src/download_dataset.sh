#!/bin/bash

DIR="/scratch/t.tovi/datasets/"

# Download the datasets
wget --directory-prefix=$DIR http://images.cocodataset.org/zips/train2017.zip
wget --directory-prefix=$DIR http://images.cocodataset.org/zips/val2017.zip
wget --directory-prefix=$DIR http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

# Unpack the datasets
mkdir -p ${DIR}/COCO_stuff_images
mkdir -p ${DIR}/COCO_stuff_annotations
unzip ${DIR}/train2017.zip -d ${DIR}/COCO_stuff_images/
unzip ${DIR}/val2017.zip -d ${DIR}/COCO_stuff_images/
unzip ${DIR}/stuffthingmaps_trainval2017.zip -d ${DIR}/COCO_stuff_annotations/
