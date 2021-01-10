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
#settings 
seq_length = 3200
batch_size = 1
#_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR =  "log"
MODEL_DIR = "ckpt"
BPE_TSV_PATH ="bpe_spm.tsv"
BPE_MODEL_PATH = "bpe_model"
DATASET_PATH = "enwik8.gz"
BOS_ID = 3
EOS_ID = 4
trsh = 5
vocab_size = 20000
learning_rate=1e-5
epoch = 100
GPU= False

tf.config.experimental.set_visible_devices([], 'GPU')


_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"
MODEL_DIR = _ROOT + "/ckpt"


FLAGS = flags.FLAGS



#settings 
seq_length = 3200
batch_size = 8
#_ROOT = os.path.abspath(os.path.dirname(__file__))

BPE_TSV_PATH ="bpe_model.model"

BPE_MODEL_PATH = "bpe_model.model"


DATASET_PATH = "enwik8.gz"
BOS_ID = 3
EOS_ID = 4
trsh = 5
vocab_size = 20000
learning_rate=5e-5


#@click.command()
#@click.option('--model-path', type=str, default="./model", show_default=True, help="Model Path")
#@click.option('--model-param', type=str, default="./model/model_par.json", show_default=True, help="Model Parm")
#@click.option('--vocab', type=str, default="./data/bpe_model.model", show_default=True, help="Vocab")
#@click.option('--seq-len', type=int, default=512, show_default=True, help="seq_len")
#@click.option('--temperature', type=float, default=1.0, show_default=True, help="seq_len")
#@click.option('--top-k', type=int, default=8, show_default=True, help="seq_len")
#@click.option('--top-p', type=float, default=0.9, show_default=True, help="seq_len")
#@click.option('--nucleus_sampling', type=bool, default=False, show_default=True, help="seq_len")
#@click.option('--context', type=str, default="sample context", show_default=True, help="Context given to model")

flags.DEFINE_integer("batch_size", 2,"batch_size")
flags.DEFINE_integer("vocab_size", 20000,"vocab_size")
flags.DEFINE_integer("embedding_size", 128,"Embedding size")
flags.DEFINE_integer("top_k", 3,"Embedding size")
flags.DEFINE_float("top_p", 0.9,"Embedding size")
flags.DEFINE_integer("seq_len", 3200,"Embedding size")
flags.DEFINE_bool("nucleus_sampling", False, "s")

flags.DEFINE_string("optimizer", "adam","optimizer type")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning")
flags.DEFINE_bool("distributed", False, "Whether to use TPU or GPU/CPU.")

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
					emb = 128,
					depth = 1,
					max_seq_len = FLAGS.seq_len,
					heads = 8,
					lsh_dropout = 0.1,
					causal = True,        # auto-regressive or not
					bucket_size = 32,     # average size of qk per bucket, 64 was recommended in paper
					n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
					ff_chunks = 16,      # number of chunks for feedforward layer, make higher if there are memory issues
					weight_tie = False,   # tie parameters of each layer for no memory per additional depth
					attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
					use_full_attn = True )  #
    sg = SequenceGenerator(model,MODEL_DIR,FLAGS)
    sg.load_weights()
    generated_seq = sg.sample_sequence(data_val[:200],
									   seq_len=FLAGS.seq_len,
									   temperature=1.0,
									   top_k=8,
									   top_p=0.9,
									   nucleus_sampling=True)

    print("Generated seq by model:- " + generated_seq)




if __name__ == '__main__':
	app.run(main)


