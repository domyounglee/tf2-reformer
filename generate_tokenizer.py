import tensorflow as tf
import numpy as np 
import gzip
from collections import Counter
import csv
import sentencepiece as spm
import os 

#_ROOT = os.path.abspath(os.path.dirname(__file__))

BPE_TSV_PATH ="bpe_spm.tsv"
BPE_MODEL_PATH = "bpe_model"

BOS_ID = 3
EOS_ID = 4
trsh = 5
vocab_size = 20000

token_dict = Counter()
with gzip.open('enwik8.gz') as file:
    readed=file.read(int(95e6)).decode("utf-8") 
    dataset = readed.lower().split()
    
    token_dict.update(dataset)
    
    
    trsh = 15
    print(len(token_dict))
    token_dict = Counter(dict(filter(lambda x: x[1] >= trsh, token_dict.items())))
    print(len(token_dict))



    print("finish token_dict")
    #write vocab as tsv
    with open(BPE_TSV_PATH, 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for word in token_dict:
            tsv_output.writerow([word, token_dict[word]])
    print("finish write bpe tsv")
    spmcmd = '--input={spm_input} --model_prefix={spm_model} --input_format=tsv --vocab_size={vocab_size} --user_defined_symbols=[SEP],[BOS],[EOS] --hard_vocab_limit=false --model_type=bpe --pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1 --pad_piece=[PAD] --unk_piece=[UNK]'.format(
        spm_input=BPE_TSV_PATH, spm_model=BPE_MODEL_PATH, vocab_size=vocab_size)
    spm.SentencePieceTrainer.train(spmcmd)
    print("finish train bpe ")
   
