#!/usr/bin/env bash

./build.sh

docker save joeranbosma/prostate_cancer_detection_processor:latest | gzip -c > prostate_cancer_detection_processor.tar.gz
