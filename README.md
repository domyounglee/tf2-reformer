# Reformer TF2

Reproduced the Revnet and LSH attention  in tf2 

## Use Docker 
    docker pull tensorflow/tensorflow:2.3.0-gpu
    ./docker_tf2.sh
Don't forget to change the mapping path 

## dataset

The enwik8 data was downloaded from the Hutter prize page: [here](http://prize.hutter1.net/)

## train

	python main.py
	
## ref

  This code is based on https://github.com/cerebroai/reformers
