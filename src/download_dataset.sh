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
# email: ben.j.mueller@wustl.edu
# username: BenMueller
# password: CityScapes12345#  (try cityscapes12345 if that doesn't work)

# note: if this is breaking, it might be because of the need for a login
    # if this is the case, need to download the file locally to your machine from the website
    # and then transfer them to the server / compute that you're using
# curl -o "$DIR/Cityscapes_gtFine.zip" "https://www.cityscapes-dataset.com/file-handling/?packageID=1"

# ADE20K dataset
# ADE20k login
# username & displayname: BenMueller
# pass: ADE20k12345#
# email: ben.j.mueller@wustl.edu

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