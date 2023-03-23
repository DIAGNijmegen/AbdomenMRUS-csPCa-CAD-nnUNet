#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build "$SCRIPTPATH" \
    -t joeranbosma/prostate_cancer_detection_processor:v2 \
    -t joeranbosma/prostate_cancer_detection_processor:latest
