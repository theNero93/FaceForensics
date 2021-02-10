#!/bin/sh

WGET_URL="http://kaldir.vc.in.tum.de/FaceForensics/models/faceforensics++_models.zip"

mkdir ./models
wget $WGET_URL -O ./models/faceforensics++_models.zip
unzip ./models/faceforensics++_models.zip -d ./models
rm ./models/faceforensics++_models.zip
mv ./models/faceforensics++_models_subset/face_detection ./models/face_detection
mv ./models/faceforensics++_models_subset/full ./models/full
rmdir ./models/faceforensics++_models_subset
