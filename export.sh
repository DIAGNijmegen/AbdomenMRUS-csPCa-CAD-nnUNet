#!/usr/bin/env bash

./build.sh

docker save prostatecancerdetectioncontainer | gzip -c > AbdomenMRUS-csPCa-nnUNet-CAD-bpMRI.tar.gz
