#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut -c 1-10)

DOCKER_FILE_SHARE=prostate_cancer_detection_processor-output-$VOLUME_SUFFIX
docker volume create $DOCKER_FILE_SHARE
# you can see your output (to debug what's going on) by specifying a path instead:
# DOCKER_FILE_SHARE="/mnt/netcache/pelvis/projects/joeran/tmp-docker-volume"

docker run --cpus=4 --memory=32gb --shm-size=32gb --gpus='"device=0"' --rm \
        -v $SCRIPTPATH/test/:/input/ \
        -v $DOCKER_FILE_SHARE:/output/ \
        joeranbosma/prostate_cancer_detection_processor

docker run --rm \
        -v $DOCKER_FILE_SHARE:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        insighttoolkit/simpleitk-notebooks:latest python -c "import sys; import numpy as np; import SimpleITK as sitk; f1 = sitk.GetArrayFromImage(sitk.ReadImage('/output/images/transverse-cancer-heatmap-prostate-mri/heatmap.mha')); f2 = sitk.GetArrayFromImage(sitk.ReadImage('/input/labels/demo001_heatmap.mha')); print('max. difference between prediction and reference:', np.abs(f1-f2).max()); sys.exit(int(np.abs(f1-f2).max() > 1e-3));"


if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm $DOCKER_FILE_SHARE

