#!/usr/bin/env bash

./build.sh

docker save prostatecancerdetectioncontainer | gzip -c > ProstateCancerDetectionContainer.tar.gz
