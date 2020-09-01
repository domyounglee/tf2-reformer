import glob
import tensorflow as tf
import os
import json
from absl import app
from absl import flags
from absl import logging
from reformers.TFreformers import TFReformerLM
from SequenceGenerator import SequenceGenerator


_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"
MODEL_DIR = _ROOT + "/ckpt"


FLAGS = flags.FLAGS



#settings 
seq_length = 3200
batch_size = 2
#_ROOT = os.path.abspath(os.path.dirname(__file__))

BPE_TSV_PATH ="bpe_spm.tsv"
BPE_MODEL_PATH = "bpe_model"
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
flags.DEFINE_integer("seq_len", 100,"Embedding size")
flags.DEFINE_bool("nucleus_sampling", False, "s")

flags.DEFINE_string("optimizer", "adam","optimizer type")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning")
flags.DEFINE_bool("distributed", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("mode", "gpt2","Language models (choose one btw gpt2 or reformer")


def main(argv):

    context = "Hello my name is Domyoung Nice to meet you "
    sg = SequenceGenerator(MODEL_DIR, BPE_TSV_PATH,FLAGS)
    sg.load_weights()
    generated_seq = sg.sample_sequence(context,
									   seq_len=FLAGS.seq_len,
									   temperature=1.0,
									   top_k=8,
									   top_p=0.9,
									   nucleus_sampling=False)
    print("Generated seq by model:- " + generated_seq)




if __name__ == '__main__':
	app.run(main)


