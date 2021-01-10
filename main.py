from reformers.TFreformers import TFReformerLM, TFLSHAttention
import tensorflow as tf
import numpy as np 
import gzip
from collections import Counter
import csv
import sentencepiece as spm
import os 
import functools
import sys 
from SequenceGenerator import SequenceGenerator
from absl import app
from absl import flags
from absl import logging
import random
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


NUM_BATCHES = int(1e5)

#settings 

#_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR =  "log"
MODEL_DIR = "ckpt"
BPE_TSV_PATH ="bpe_spm.tsv"
BPE_MODEL_PATH = "bpe_model"
DATASET_PATH = "enwik8.gz"

trsh = 5
learning_rate=1e-4
epoch = int(1e5)
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 1000

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 4,"batch_size")
flags.DEFINE_integer("vocab_size", 256,"vocab_size")
flags.DEFINE_integer("embedding_size", 128,"Embedding size")
flags.DEFINE_integer("top_k", 3,"Embedding size")
flags.DEFINE_float("top_p", 0.9,"Embedding size")
flags.DEFINE_integer("seq_len", 4096,"Embedding size")
flags.DEFINE_bool("nucleus_sampling", False, "s")

flags.DEFINE_string("optimizer", "adam","optimizer type")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning")
flags.DEFINE_bool("distributed", False, "Whether to use TPU or FLAGS.distributed/CPU.")

flags.DEFINE_string("mode", "gpt2","Language models (choose one btw gpt2 or reformer")


class TextSamplerDataset():
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = np.random.randint(0,len(self.data) - self.seq_len - 1, (1,))[0]
        
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1]
        return full_seq

    def __len__(self):
        return len(self.data) // self.seq_len


        
def main(argv):
    if FLAGS.distributed==True:
        mirrored_strategy = tf.distribute.MirroredStrategy()

    with gzip.open(DATASET_PATH) as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        print(X[:10])
        data_train, data_val = tf.convert_to_tensor(trX,dtype=tf.uint8), tf.convert_to_tensor(vaX,dtype=tf.int32)
        print(data_train[:100])

    """
    with gzip.open(DATASET_PATH) as file:
        readed=file.read(int(95e6)).decode("utf-8") 
        dataset = readed.lower().split()
    """ 
    sampler_dataset = TextSamplerDataset(data_train,FLAGS.seq_len)
    sampler_dataset_val = TextSamplerDataset(data_val,FLAGS.seq_len)
    #s = spm.SentencePieceProcessor()
    #s.Load(BPE_MODEL_PATH + ".model")


    print("finish setting datset")
    def generator_fn(dataset ):
        for i in range(len(dataset)):

            encoded_id =dataset[i]
            inputs = encoded_id[ :-1]
            targets = encoded_id[ 1:]

            yield inputs,targets      
    
    if FLAGS.distributed:
        with mirrored_strategy.scope():
            d = tf.data.Dataset.from_generator( \
                        functools.partial(generator_fn, dataset=sampler_dataset), \
                        output_types=(tf.uint8, tf.uint8), output_shapes=([FLAGS.seq_len],[FLAGS.seq_len]))
            d=d.batch(FLAGS.batch_size)
            d=d.prefetch(10)
            d = mirrored_strategy.experimental_distribute_dataset(d)


            d_val = tf.data.Dataset.from_generator( \
                        functools.partial(generator_fn, dataset=sampler_dataset_val), \
                        output_types=(tf.uint8, tf.uint8), output_shapes=([FLAGS.seq_len],[FLAGS.seq_len]))
            d_val=d_val.batch(FLAGS.batch_size)
            d_val=d_val.prefetch(10)
            d_val = mirrored_strategy.experimental_distribute_dataset(d_val)

    else:
        d = tf.data.Dataset.from_generator( \
                    functools.partial(generator_fn, dataset=sampler_dataset), \
                    output_types=(tf.uint8, tf.uint8), output_shapes=([FLAGS.seq_len],[FLAGS.seq_len]))
        d=d.batch(FLAGS.batch_size)
        d=d.prefetch(10)

        d_val = tf.data.Dataset.from_generator( \
                    functools.partial(generator_fn, dataset=sampler_dataset_val), \
                    output_types=(tf.uint8, tf.uint8), output_shapes=([FLAGS.seq_len],[FLAGS.seq_len]))
        d_val=d_val.batch(FLAGS.batch_size)
        d_val=d_val.prefetch(10)

    if FLAGS.distributed :
        with mirrored_strategy.scope():
            model_tf = TFReformerLM(
                num_tokens= FLAGS.vocab_size,
                emb = 512,
                depth = 6,   # batch 4 full attention 8 이면 안돌아감 
                max_seq_len = FLAGS.seq_len,
                heads = 8,
                lsh_dropout = 0.1,
                causal = True,        # auto-regressive or not
                bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
                n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
                ff_chunks = 10,      # number of chunks for feedforward layer, make higher if there are memory issues
                weight_tie = True,   # tie parameters of each layer for no memory per additional depth
                attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
                use_full_attn = False   # use full self attention, for comparison
            
            )
            #training settings 
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True, reduction='none', name='loss')
            accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy(
                        name='accuracy')
            train_loss = tf.keras.metrics.Mean(name='train_loss')

            model_tf.set_optimizer(tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999,
                                                                    epsilon=1e-8))
            model_tf.create_checkpoint_manager(MODEL_DIR, max_to_keep=5, load_model=False)

    else:
        model_tf = TFReformerLM(
            num_tokens= FLAGS.vocab_size,
            emb = 512,
            depth = 6,   # batch 4 full attention 8 이면 안돌아감 
            max_seq_len = FLAGS.seq_len,
            heads = 8,
            lsh_dropout = 0.1,
            causal = True,        # auto-regressive or not
            bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
            n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
            ff_chunks = 2,      # number of chunks for feedforward layer, make higher if there are memory issues
            weight_tie = True,   # tie parameters of each layer for no memory per additional depth
            attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
            use_full_attn = False   # use full self attention, for comparison
        
        )


        #training settings 
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True, reduction='none', name='loss')
        accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy(
                    name='accuracy')
        train_loss = tf.keras.metrics.Mean(name='train_loss')

        model_tf.set_optimizer(tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999,
                                                                epsilon=1e-8))
        model_tf.create_checkpoint_manager(MODEL_DIR, max_to_keep=5, load_model=False)

    sg = SequenceGenerator(model_tf, MODEL_DIR,FLAGS)#initialize model generator

    print("start training")



    if FLAGS.distributed:
        with mirrored_strategy.scope():

            for e in range(1,epoch+1):
                for (step, (inputs, targets)) in enumerate(d):
                    
                    step = ( e  ) * (step+1)

                    #y = model_tf(inputs)
                    #loss = tf.reduce_mean(get_loss(targets, y ,loss_object))
                    #print(loss)
                    #print(y)

                    loss = model_tf.train_step(inputs,targets,loss_object,train_loss,mirrored_strategy,distributed=True)
                    tf.print(step,loss)
                    
                    if step % 1000 == 0:
                        ckpt_save_path = model_tf.ckpt_manager.save()
                        print('Saving checkpoint for step {} at {}'.format(step,
                                                                            ckpt_save_path))

                    if step % VALIDATE_EVERY == 0:
                        total_eval_loss=0
                        for (eval_step, (inputs_val, targets_val)) in enumerate(d_val):
                            if eval_step==30:
                                break
                            eval_loss = model_tf.eval_step(inputs_val,targets_val,loss_object,train_loss,mirrored_strategy,distributed=True)
                            total_eval_loss+=eval_loss

                        print("eval loss",total_eval_loss/float(eval_step+1))

                    if step % GENERATE_EVERY == 0:
                        print("generate")
                        asdf = sampler_dataset_val[0][:-1]
                        print(asdf)
                        generated_seq = sg.sample_sequence(asdf,
                                                        predict_len=30,
                                                        temperature=1.0,
                                                        top_k=8,
                                                        top_p=0.9,
                                                        nucleus_sampling=True)

                        print("Generated seq by model:- " + generated_seq)     

                if step>NUM_BATCHES:
                    break




    else:
        for e in range(1,epoch+1):
            for (step, (inputs, targets)) in enumerate(d):
                step = ( e  ) * (step+1)
                #y = model_tf(inputs)
                #loss = tf.reduce_mean(get_loss(targets, y ,loss_object))
                #print(loss)
                #print(y)
                loss = model_tf.train_step(inputs,targets,loss_object,train_loss,distributed=False)
                tf.print(step,loss)       

                if step % 1000 == 0:
                    ckpt_save_path = model_tf.ckpt_manager.save()
                    print('Saving checkpoint for step {} at {}'.format(step,
                                                                        ckpt_save_path))
                if step % VALIDATE_EVERY == 0:
                    total_eval_loss=0
                    for (eval_step, (inputs_val, targets_val)) in enumerate(d_val):
                        if eval_step==30:
                            break
                        eval_loss = model_tf.eval_step(inputs_val,targets_val,loss_object,train_loss,distributed=False)
                        total_eval_loss+=eval_loss

                    print("eval loss",total_eval_loss/float(eval_step+1))

                if step % GENERATE_EVERY == 0:
                    print("generate")
                    asdf = sampler_dataset_val[0][:-1]
                    #print(asdf)
                    generated_seq = sg.sample_sequence(asdf,
                                                    predict_len=30,
                                                    temperature=1.0,
                                                    top_k=8,
                                                    top_p=0.9,
                                                    nucleus_sampling=True)

                    print("Generated seq by model:- " + generated_seq)     

    



            if step>NUM_BATCHES:
                break

if __name__ == '__main__':
	app.run(main)
