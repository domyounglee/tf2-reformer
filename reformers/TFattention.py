# MIT License

# Copyright (c) 2020 Streack, Jayakrishna Sahit

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import tensorflow as tf
from tensorflow.keras.layers import Dense

def mask_fill_inf(matrix, mask):
    negmask = 1 - mask
    num = 3.4 * math.pow(10, 38)
    return (matrix * mask) + (-((negmask * num + num) - num))

class MultiHeadAttention(tf.keras.Model):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.query_dense = Dense(units=d_model)
        self.key_dense = Dense(units=d_model)
        self.value_dense = Dense(units=d_model)
        self.dense = Dense(units=d_model)

    def merge_heads(self,v,batch_size):
        return tf.reshape(tf.transpose(tf.reshape(v, (batch_size, -1, self.num_heads, self.depth)), perm=[0, 2, 1, 3]), (batch_size , -1, self.d_model)) 

    def split_heads(self,v,batch_size):
        return tf.transpose(tf.reshape(v, (batch_size, -1, self.num_heads, self.depth)), perm=[0, 2, 1, 3])


    def scaled_dot_product(self, q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = scaled_attention_logits / tf.math.sqrt(dk)


        def create_look_ahead_mask(size):
            mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
            return mask  # (seq_len, seq_len) 
        mask = create_look_ahead_mask(tf.shape(k)[-2])
      
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)


        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        attention_weights = tf.nn.dropout(attention_weights, rate=0.1, name="attn_dropout")
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        

        return output, attention_weights

    def call(self, inputs):
        query, key, value = inputs['query'], inputs['key'], inputs[
            'value']
        batch_size = tf.shape(query)[0]
        dim = tf.shape(query)[-1]
        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        output,attention_map = self.scaled_dot_product(query, key, value)
        #scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        #concat_attention = tf.reshape(scaled_attention,
        #                            (batch_size, -1, self.d_model))
        scaled_attention = self.merge_heads(output,batch_size)
        outputs = self.dense(scaled_attention)
        return outputs


class TFSelfAttention(tf.keras.Model):
    def __init__(self, emb, heads = 8, causal = False):
        super().__init__()
        assert emb % heads == 0, 'dimensions must be divisible by number of heads'
        self.attn = MultiHeadAttention(emb, heads)
        self.to_out = Dense(emb)
        self.causal = causal

    def call(self, inputs):
        b, t, e = inputs.shape


        output = self.attn({'query' : inputs, 'key' : inputs, 'value' : inputs})
        return self.to_out(output)


class TFFeedForward(tf.keras.Model):
    def __init__(self, emb, mult = 4):
        super().__init__()
        self.emb = emb
        self.proj_in = Dense(emb * mult)
        self.proj_out = Dense(emb)

    def call(self, inputs):
        #tf.print("fdafdsaf")
        #tf.print(inputs.shape)
        inputs = self.proj_in(inputs)
        inputs = tf.keras.activations.relu(inputs)
        inputs = self.proj_out(inputs)
        return inputs