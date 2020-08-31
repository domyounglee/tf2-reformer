# Reformer TF2
(under construction)
This code is based on [https://github.com/cerebroai/reformers](https://github.com/cerebroai/reformers)

## Use Docker 
    docker pull tensorflow/tensorflow:2.3.0-gpu
    ./docker_tf2.sh
Don't forget to change the mapping path 

### requirements
    pip install -r requirements.txt
You need only *sentencepiece* packeage 

### dataset

The enwik8 data was downloaded from the Hutter prize page: [here](http://prize.hutter1.net/)

### train

	python main.py
	

