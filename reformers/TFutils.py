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

import tensorflow as tf
from tensorflow.keras import layers


def get_padding(x, padding_value=0, dtype=tf.float32):
    """Return float tensor representing the padding values in x.
    Args:
    x: int tensor with any shape
    padding_value: int value that
    dtype: The dtype of the return value.
    Returns:
    float tensor with same shape as x containing values 0 or 1.
    0 -> non-padding, 1 -> padding
    """
    return tf.cast(tf.equal(x, padding_value), dtype)

def make_unit_length(x, epsilon=1e-6):
    norm = tf.norm(x,  ord=2, axis=-1, keepdims=True)
    return tf.math.truediv(x, norm + epsilon)
def sort_key_val(t1, t2, axis=-1):

    values = tf.sort(t1, axis=axis)
    
    offset=tf.range(t1.shape[0])*t1.shape[1]
    offset=tf.reshape(offset,[-1,1])
    offset=tf.broadcast_to(offset, t1.shape)

    t2 = tf.broadcast_to(t2, t1.shape)


    return values, tf.gather(tf.reshape(t2,[-1]), tf.argsort(t1, axis=axis)+offset, axis=axis)

def batched_index_select(values, indices):

    seq_len = values.shape[1]
    last_dim = values.shape[-1]

    offset=tf.range(indices.shape[0])*seq_len
    offset=tf.reshape(offset,[-1,1])
    offset=tf.broadcast_to(offset, indices.shape)
    
    flatten_values = tf.reshape(values,[-1,last_dim])
    return tf.gather(flatten_values, indices+offset)

def process_inputs_chunk(fn, *args, chunks=1):
    chunked_inputs = list(map(lambda x: tf.split(x, chunks, axis=0), args))
    outputs = [fn(*input_pair) for input_pair in zip(*chunked_inputs)] #chunking 된 q ,kv 끼리 묶여서 input_pair를 만든다. 
    return outputs

def cache_fn(f):
    cache = None
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class ScaleNorm(layers.Layer):
    def __init__(self, emb, eps):
        super(ScaleNorm, self).__init__()
        self.g = tf.Variable(initial_value=w_init(shape=(1,),
                        dtype='float32'),
                        trainable=True)
        self.eps = eps

    def call(self, inputs):
        n = tf.norm(inputs, axis=-1, keepdims=True).clip_by_value(min=self.eps)
        return x / n * self.g

class WithNorm(layers.Layer):
    def __init__(self, norm_class, emb, fn):
        super(WithNorm, self).__init__()
        self.emb = emb
        if isinstance(norm_class, ScaleNorm):
            self.norm = norm_class(emb)
        else:
            self.norm = norm_class()

        self.fn = fn

    def call(self, inputs,input_padding_mask=None):
        inputs = self.norm(inputs)
        if input_padding_mask is not None:
            return self.fn(inputs,input_padding_mask)
        else:
            return self.fn(inputs)

class Chunk(layers.Layer):
    def __init__(self, chunks, fn, along_axis = -1):
        super(Chunk, self).__init__()
        self.axis = along_axis
        self.chunks = chunks
        self.fn = fn

    def call(self, inputs):

        chunks = tf.split(inputs, self.chunks, axis= self.axis)
        #tf.print("chunk")
        #tf.print(inputs.shape)
        #tf.print(chunks[0].shape)
        #tf.print(len(chunks))
        return tf.concat([self.fn(c) for c in chunks], axis = self.axis)