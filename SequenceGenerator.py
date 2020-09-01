import tensorflow as tf
import sentencepiece as spm
import json

from reformers.TFreformers import TFReformerLM

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
    indices = tf.stack([
        tf.range(0, batch),
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


class SequenceGenerator:

    def __init__(self, model_path, vocab_path,flags):
        self.sp = None
        self.model = None
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.flags=flags

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
        self.sp.load(self.vocab_path)

    def sample_sequence(self,
                        context=None,
                        seq_len=512,
                        bos=3,
                        eos=4,
                        temperature=1,
                        top_k=8,
                        top_p=8,
                        nucleus_sampling=True):

        if context == None:
            print("Give some context to model.................")
            return
        context = tf.expand_dims(([bos] + self.sp.encode_as_ids(context)), 0)
        prev = context
        output = context
   
        for i in range(seq_len):
            logits= self.model(prev, training=False,)
            # print(logits)
            logits = logits[:, -1, :] / tf.cast(temperature, tf.float32)
            # print(logits)
            logits = top_k_logits(logits, k=top_k)
            # print(logits)
            if nucleus_sampling:
                logits = top_p_logits(logits, p=top_p)

            samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
            # print(samples)
            if tf.equal(samples, eos):
                # print("Predicted end of sequence.")
                break

            # print("shape.........")
            # print(tf.shape(output))
            # print(tf.shape(samples))
            output = tf.concat([output, samples], axis=-1)
            prev = samples
            # print(tf.shape(output))
            # print(output)

        # print("--------------------------")
        result = tf.squeeze(output, axis=0)
        pred = [int(i) for i in result]
        generated_seq = self.sp.decode_ids(pred[1:])
        generated_seq = generated_seq.replace("[SEP]", "").strip()
        generated_seq = ' '.join(generated_seq.split())
        return generated_seq
