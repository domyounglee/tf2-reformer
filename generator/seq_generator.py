import glob
import tensorflow as tf
import os
import json
from absl import app
from absl import flags
from absl import logging
from reformers.TFreformers import TFReformerLM
from SequenceGenerator import SequenceGenerator
from reformers.TFreformers import TFReformerLM, TFLSHAttention
import numpy as np 
import gzip
from collections import Counter
import csv
import sentencepiece as spm
import os 
import functools
import sys 

DATASET_PATH = "enwik8.gz"

tf.config.experimental.set_visible_devices([], 'GPU')


_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"
MODEL_DIR = _ROOT + "/ckpt"


FLAGS = flags.FLAGS



flags.DEFINE_integer("batch_size", 4,"batch_size")
flags.DEFINE_integer("vocab_size", 256,"vocab_size")
flags.DEFINE_integer("embedding_size", 512,"Embedding size")
flags.DEFINE_integer("top_k", 3,"Embedding size")
flags.DEFINE_float("top_p", 0.9,"Embedding size")
flags.DEFINE_integer("seq_len", 3072,"Embedding size")
flags.DEFINE_bool("nucleus_sampling", True, "s")

flags.DEFINE_string("optimizer", "adam","optimizer type")
flags.DEFINE_float("learning_rate", 1e-4, "The initial learning")
flags.DEFINE_bool("distributed", False, "Whether to use TPU or FLAGS.distributed/CPU.")

flags.DEFINE_string("mode", "gpt2","Language models (choose one btw gpt2 or reformer")


def main(argv):
    with gzip.open(DATASET_PATH) as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        print(X[:10])
        data_train, data_val = tf.convert_to_tensor(trX,dtype=tf.uint8), tf.convert_to_tensor(vaX,dtype=tf.int32)
    context = "nice to meet you. what kind of food do you like?"
    model = TFReformerLM(
					num_tokens= FLAGS.vocab_size,
					emb = FLAGS.embedding_size,
					depth = 6,
					max_seq_len = FLAGS.seq_len,
					heads = 8,
					lsh_dropout = 0.1,
					causal = True,        # auto-regressive or not
					bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
					n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
					ff_chunks = 8,      # number of chunks for feedforward layer, make higher if there are memory issues
					weight_tie = True,   # tie parameters of each layer for no memory per additional depth
					attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
					use_full_attn = True )  #
    sg = SequenceGenerator(model,MODEL_DIR,FLAGS)
    sg.load_weights()
    generated_seq = sg.sample_sequence(data_val[:200],
								
									   predict_len=30,
									   temperature=1.0,
									   top_k=8,
									   top_p=0.9,
									   nucleus_sampling=True,
									   FLAGS=FLAGS)

    print("Generated seq by model:- " + generated_seq)




if __name__ == '__main__':
	app.run(main)


