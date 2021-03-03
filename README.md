# Reformer TF2

Reproduced the [Reformer](https://arxiv.org/abs/2001.04451) in tf2 

## Dependancy 

Tensorflow >= 2.3.0

## Usage

### use Docker >= 19.03

    docker pull tensorflow/tensorflow:2.3.0-gpu
    ./docker_tf2.sh
Don't forget to change the mapping path with your own directory

### download enwik8 dataset

The enwik8 data was downloaded from the Hutter prize page: [here](http://prize.hutter1.net/)

Unicode code points are directly used as vocabulary index 

### set hyperprameters 
```
model_tf = TFReformerLM(
    num_tokens= FLAGS.vocab_size,
    emb = FLAGS.embedding_size,
    depth = 6,   # batch 4 full attention 8 이면 안돌아감 
    max_seq_len = FLAGS.seq_len,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True,        # auto-regressive or not
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    ff_chunks = 8,      # number of chunks for feedforward layer, make higher if there are memory issues
    weight_tie = True,   # tie parameters of each layer for no memory per additional depth
    attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    use_full_attn = False   # use full self attention, for comparison

)
```

You only need Tesla V100 of 16G to run the reformer model with above settings on 4096 sequence length and batch size of 4 

### train 
```
python main.py
```

the main.py code includes generation code and you can generate the sequence with generator/seq_generator.py 

## Reference

  This code is based on 
  
  https://github.com/akanyaani/gpt-2-tensorflow2.0

  https://github.com/cerebroai/reformers

  https://github.com/lucidrains/reformer-pytorch