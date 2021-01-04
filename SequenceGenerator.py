import tensorflow as tf
import sentencepiece as spm
import json
import math
from reformers.TFreformers import TFReformerLM

def pad_to_multiple(tensor, seqlen, multiple, dim=-1):
    m = seqlen / multiple
    tf.print("flag")
    tf.print(m)
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return tf.pad(tensor, [[0,0] , [0,remainder]], constant_values=0)


def argmax(logits):
    return tf.argmax(logits)


def top_k_logits(logits, k):
    if k == 0:
        return logits

    values, _ = tf.nn.top_k(logits, k=k)
    min_values = values[:, -1]

    return tf.where(
        logits < min_values,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits
    )

    # Nucleas Sampling (https://arxiv.org/pdf/1904.09751.pdf)


def top_p_logits(logits, p):
    """Took from OpenAI GPT-2 Implememtation"""
    batch = tf.shape(logits)[0]
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    tf.print(cumulative_probs)
    indices = tf.stack([
        tf.range(0, batch),
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    tf.print(indices.shape)
    tf.print(indices)
    tf.print(min_values.shape)
    tf.print(sorted_logits.shape)

    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


class SequenceGenerator:

    def __init__(self, model_path, spmodel_path,flags):
        
        self.sp = None
        self.model = None
        self.model_path = model_path
        self.spmodel_path = spmodel_path
        self.flags=flags
        print(self.flags.vocab_size)
        print(self.flags.seq_len)

    def load_weights(self):

        self.model = TFReformerLM(
                num_tokens= self.flags.vocab_size,
                emb = 128,
                depth = 1,
                max_seq_len = self.flags.seq_len,
                heads = 8,
                lsh_dropout = 0.1,
                causal = True,        # auto-regressive or not
                bucket_size = 32,     # average size of qk per bucket, 64 was recommended in paper
                n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
                ff_chunks = 16,      # number of chunks for feedforward layer, make higher if there are memory issues
                weight_tie = False,   # tie parameters of each layer for no memory per additional depth
                attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
                use_full_attn = False )  # use full self attention, for comparison
            
        ckpt = tf.train.Checkpoint(model=self.model)

        ckpt_manager = tf.train.CheckpointManager(ckpt, self.model_path, max_to_keep=1)

        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Model weights loaded into memory')

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.spmodel_path)

    def sample_sequence(self,
                        context=None,
                        seq_len=512,
                        bos=3,
                        eos=4,
                        temperature=1,
                        top_k=8,
                        top_p=0.9,
                        nucleus_sampling=True):

        if context == None:
            print("Give some context to model.................")
            return
        context = tf.expand_dims(([bos] + self.sp.encode_as_ids(context)), 0)
        tf.print(context.shape)
        t = context.shape[1]
        
        tf.print(context)
        output = context
        max_window = 128
        for i in range(seq_len):
            tf.print("start")
            print(output)
            padded_output = pad_to_multiple(output, output.shape[1] ,64)
            tf.print(padded_output.shape)
            tf.print(padded_output[:,-max_window:])

            logits= self.model(padded_output[:,-max_window:], training=False,)

            pred_i = t+i+1
            logits = logits[:,:pred_i]
            print(logits)
            logits = logits[:, -1, :] / tf.cast(temperature, tf.float32)
            print(logits)

            if nucleus_sampling:
                logits = top_p_logits(logits, p=top_p)
            else: #topk
                logits = top_k_logits(logits, k=top_k)

            samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
            # print(samples)
            if tf.equal(samples, eos):
                # print("Predicted end of sequence.")
                break

            # print("shape.........")   
            # print(tf.shape(output))
            # print(tf.shape(samples))
            output = tf.concat([output, samples], axis=-1)
            print("end")
            print(output.shape)
            #prev = output
            # print(tf.shape(output))
            # print(output)

        # print("--------------------------")
        result = tf.squeeze(output, axis=0)
        pred = [int(i) for i in result]
        generated_seq = self.sp.decode_ids(pred[1:])
        generated_seq = generated_seq.replace("[SEP]", "").strip()
        generated_seq = ' '.join(generated_seq.split())
        return generated_seq
