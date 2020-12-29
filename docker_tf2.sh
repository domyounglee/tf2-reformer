#! /bin/bash
docker run -u 0 --net=host --gpus 0,1,2,4 \
 --privileged \
 -v /data1/dmlee/reformer/:/workspace/ \
 -it -p 8899:8899 --name tf2  -e LC_ALL=C.UTF-8 -e DISPLAY=$DISPLAY \
 tensorflow/tensorflow:2.3.0-gpu \
 /bin/bash
