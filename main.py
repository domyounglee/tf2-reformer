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
mirrored_strategy = tf.distribute.MirroredStrategy()
#settings 
seq_length = 3200
batch_size = 4
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
GPU= True

        
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

with gzip.open(DATASET_PATH) as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    print(X[:10])
    data_train, data_val = tf.convert_to_tensor(trX,dtype=tf.uint8), tf.convert_to_tensor(vaX,dtype=tf.uint8)
print(data_train[:100])

"""
with gzip.open(DATASET_PATH) as file:
    readed=file.read(int(95e6)).decode("utf-8") 
    dataset = readed.lower().split()
""" 
sampler_dataset = TextSamplerDataset(data_train,seq_length)
#s = spm.SentencePieceProcessor()
#s.Load(BPE_MODEL_PATH + ".model")

print("finish setting datset")
def generator_fn(dataset, bs,seq_len ):
    for i in range(len(dataset)):

        encoded_id =sampler_dataset[i]
        inputs = encoded_id[ :-1]
        targets = encoded_id[ 1:]

        yield inputs,targets
        
 
if GPU:
    with mirrored_strategy.scope():
        d = tf.data.Dataset.from_generator( \
                    functools.partial(generator_fn, dataset=sampler_dataset, bs=batch_size,seq_len=seq_length), \
                    output_types=(tf.uint8, tf.uint8), output_shapes=([seq_length],[seq_length]))
        d=d.batch(batch_size)
        d=d.prefetch(10)
        d = mirrored_strategy.experimental_distribute_dataset(d)

else:
    d = tf.data.Dataset.from_generator( \
                functools.partial(generator_fn, dataset=sampler_dataset, bs=batch_size,seq_len=seq_length), \
                output_types=(tf.int64, tf.int64), output_shapes=([seq_length],[seq_length]))
    d=d.batch(batch_size)
    d=d.prefetch(10)
    

if GPU :
    with mirrored_strategy.scope():
        model_tf = TFReformerLM(
            num_tokens= vocab_size,
            emb = 128,
            depth = 6,   # batch 4 full attention 8 이면 안돌아감 
            max_seq_len = seq_length,
            heads = 8,
            lsh_dropout = 0.1,
            causal = True,        # auto-regressive or not
            bucket_size = 32,     # average size of qk per bucket, 64 was recommended in paper
            n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
            ff_chunks = 16,      # number of chunks for feedforward layer, make higher if there are memory issues
            weight_tie = False,   # tie parameters of each layer for no memory per additional depth
            attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
            use_full_attn = True   # use full self attention, for comparison
        
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
    num_tokens= vocab_size,
    emb = 128,
    depth = 4,
    max_seq_len = seq_length,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True,        # auto-regressive or not
    bucket_size = 32,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    ff_chunks = 16,      # number of chunks for feedforward layer, make higher if there are memory issues
    weight_tie = False,   # tie parameters of each layer for no memory per additional depth
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

print("start training")

if GPU:
    with mirrored_strategy.scope():

        for e in range(1,epoch+1):
            for (step, (inputs, targets)) in enumerate(d):
                
                step = ( e  ) * (step+1)
                #y = model_tf(inputs)
                #loss = tf.reduce_mean(get_loss(targets, y ,loss_object))
                #print(loss)
                #print(y)

                loss = model_tf.train_step(inputs,targets,loss_object,train_loss,mirrored_strategy,GPU=True)
                print(loss)
                
                if step % 1000 == 0:
                    ckpt_save_path = model_tf.ckpt_manager.save()
                    print('Saving checkpoint for step {} at {}'.format(step,
                                                                        ckpt_save_path))

else:
    for e in range(1,epoch+1):
        for (step, (inputs, targets)) in enumerate(d):
            step = ( e  ) * (step+1)
            #y = model_tf(inputs)
            #loss = tf.reduce_mean(get_loss(targets, y ,loss_object))
            #print(loss)
            #print(y)
            loss = model_tf.train_step(inputs,targets,loss_object,train_loss)
            print(loss)
            
            if step % 10000 == 0:
                ckpt_save_path = model_tf.ckpt_manager.save()
                print('Saving checkpoint for step {} at {}'.format(step,
                                                                    ckpt_save_path))



