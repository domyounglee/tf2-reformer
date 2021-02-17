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

def process_inputs_chunk(fn, *args, seed_, chunks=1):
    chunked_inputs = list(map(lambda x: tf.split(x, chunks, axis=0), args))
    outputs = [fn(*input_pair, seed_) for i,input_pair in enumerate(zip(*chunked_inputs))] #chunking 된 q ,kv 끼리 묶여서 input_pair를 만든다. 
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
    def __init__(self, chunks, fn, along_axis = -2):
        super(Chunk, self).__init__()
        self.axis = along_axis
        self.chunks = chunks
        self.fn = fn

    def call(self, inputs, seed_):

        chunks = tf.split(inputs, self.chunks, axis= self.axis)
        #tf.print("chunk")
        #tf.print(inputs.shape)
        #tf.print(chunks[0].shape)
        #tf.print(len(chunks))
        return tf.concat([self.fn(c, seed_) for i,c in enumerate(chunks)], axis = self.axis)

import torch
from torch import nn
from operator import mul
from functools import reduce

class TF_AxialPositionalEmbedding(layers.Layer):
    def __init__(self, dim, axial_shape, axial_dims = None):
        super().__init__()

        self.dim = dim
        self.shape = axial_shape
        self.max_seq_len = reduce(mul, axial_shape, 1)

        self.summed = axial_dims is None
        axial_dims = ((dim,) * len(axial_shape)) if self.summed else axial_dims

        assert len(self.shape) == len(axial_dims), 'number of axial dimensions must equal the number of dimensions in the shape'
        assert self.summed or not self.summed and sum(axial_dims) == dim, f'axial dimensions must sum up to the target dimension {dim}'

        self.weights_ = ParameterList(self, 'weights', len(axial_shape))

        for ind, (shape, axial_dim) in enumerate(zip(self.shape, axial_dims)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, axial_dim)
            tf.print(ax_shape)





            initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
            init_values = initializer(shape=(2, 2))
            ax_emb = tf.Variable(
                                name=f"ax_emb_{ind}", 
                                 initial_value=init_values, 
                                 dtype=tf.float32
                                )

            #ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0, 1))
            self.weights_.append(ax_emb)

    def forward(self, x):
        b, t, e = x.shape
        assert (t <= self.max_seq_len), f'Sequence length ({t}) must be less than the maximum sequence length allowed ({self.max_seq_len})'
        embs = []

        for ax_emb in self.weights_.to_list():
            axial_dim = ax_emb.shape[-1]
            expand_shape = (b, *self.shape, axial_dim)
            #emb = ax_emb.expand(expand_shape).reshape(b, self.max_seq_len, axial_dim)
            emb = tf.reshape(tf.tile(ax_emb,expand_shape),(b, self.max_seq_len, axial_dim))
            embs.append(emb)

        pos_emb = sum(embs) if self.summed else tf.concat(embs, axis=-1)
        return pos_emb[:, :t]

# a mock parameter list object until below issue is resolved
# https://github.com/pytorch/pytorch/issues/36035
class ParameterList(object):
    def __init__(self, kls, prefix, length):
        self.ind = 0
        self.kls = kls
        self.prefix = prefix
        self.length = length

    def _keyname(self, prefix, ind):
        return f'{prefix}_{ind}'

    def append(self, x):
        setattr(self.kls, self._keyname(self.prefix, self.ind), x)
        self.ind += 1

    def to_list(self):
        return [getattr(self.kls, self._keyname(self.prefix, i)) for i in range(self.length)]

