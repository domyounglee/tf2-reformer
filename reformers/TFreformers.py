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
from tensorflow.keras.layers import Embedding, LayerNormalization, Dense
from .TFefficient_attention import TFLSHAttention, TFLSHSelfAttention
from .TFattention import TFSelfAttention, TFFeedForward
from .TFutils import cache_fn, Chunk, WithNorm
from .blocks import ReversibleBlock, ReversibleSequence

class TFReformer(tf.keras.Model):
    def __init__(self, emb, depth, max_seq_len, heads = 8, bucket_size = 64, n_hashes = 8, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., lsh_attend_across_buckets = False, lsh_allow_duplicate_attention = True, random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, use_full_attn = False):
        super().__init__()
        self.emb = emb
        self.depth = depth

        get_full_attn = lambda: TFSelfAttention(emb, heads, causal = causal)
        get_lsh_attn = lambda: TFLSHSelfAttention(emb, heads, bucket_size, n_hashes, causal = causal, dropout = lsh_dropout, attn_chunks = attn_chunks, allow_duplicate_attention = lsh_allow_duplicate_attention, attend_across_buckets = lsh_attend_across_buckets, random_rotations_per_head = random_rotations_per_head)

        get_attn = get_full_attn if use_full_attn else get_lsh_attn
        get_ff = lambda: TFFeedForward(emb)

        if weight_tie:
            get_attn = cache_fn(get_attn)
            get_ff = cache_fn(get_ff)

        blocks = []
        norm_type = ScaleNorm if use_scale_norm else LayerNormalization

        for _ in range(depth):
            attn = get_attn()
            parallel_net = get_attn() if twin_attention else get_ff()
            f = WithNorm(norm_type, emb, attn)
            g = WithNorm(norm_type, emb, parallel_net)

            if not twin_attention and ff_chunks > 1:
                g = Chunk(ff_chunks, g, along_axis = -2)

            blocks.append(ReversibleBlock(f, g, split_along_axis=-1))

        self.model_layers = ReversibleSequence(blocks)

    def call(self, x):
        x = tf.concat([x, x], axis = -1) #revnet
        x = self.model_layers(x)
        return tf.stack(tf.reduce_sum(tf.split(x, 2, axis=-1), axis=0))

class TFReformerLM(tf.keras.Model):
    def __init__(self, num_tokens, emb, depth, max_seq_len, heads = 8, bucket_size = 64, n_hashes = 8, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, use_full_attn = False):
        super().__init__()
        self.token_emb = Embedding(num_tokens, emb)
        self.pos_emb = Embedding(max_seq_len, emb)
        self.reformer = TFReformer(emb, depth, max_seq_len, heads = heads,lsh_attend_across_buckets = True, bucket_size = bucket_size, n_hashes = n_hashes, ff_chunks = ff_chunks, attn_chunks = attn_chunks, causal = causal, weight_tie = weight_tie, lsh_dropout = lsh_dropout, random_rotations_per_head = random_rotations_per_head, twin_attention = twin_attention, use_scale_norm = use_scale_norm, use_full_attn = use_full_attn)
        self.to_logits = Dense(num_tokens)
        self.reformer_output=None
    def call(self, inputs):
        
        inputs = self.token_emb(inputs) + self.pos_emb(tf.range(inputs.shape[1]))
        inputs = self.reformer(inputs)
        self.reformer_output = inputs
        return self.to_logits(inputs)

    @tf.function
    def train_step(self,inputs,targets,loss_object,training=True):
        grads_all = []
        vars_all = []
        def get_loss(real, pred, loss_object):
            with tf.name_scope("loss_layer"):
                mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 0)),tf.float32)
                loss_ = loss_object(real, pred)
                loss_ = tf.reduce_sum(loss_, axis=1)
                sequence_avg_loss = loss_ / tf.reduce_sum(mask, axis=1)
                
                return sequence_avg_loss
        """
        with tf.GradientTape(persistent=True) as tape:
            
            tape.watch(self.to_logits.variables)
            tape.watch(self.reformer_output)

            variables_names = [v.name for v in self.reformer.model_layers.blocks[0].f.trainable_variables]
            
            tf.print(variables_names)
            tf.print(self.reformer.model_layers.blocks[0].f.trainable_variables)
            y_hat = self.call(inputs)
            loss = tf.reduce_mean(get_loss(targets, y_hat ,loss_object))
        """
        y_hat = self.call(inputs)
        loss = tf.reduce_mean(get_loss(targets, y_hat ,loss_object))
  
        dense_grad = tf.gradients(loss, self.to_logits.variables)
        grads_all.append(dense_grad)
        vars_all.append(self.to_logits.variables)
        reformer_output_grad =  tf.gradients(loss, self.reformer_output)
        tf.print("hello")
        tf.print(type(reformer_output_grad))
        tf.print(type(dense_grad))
        tf.print(tf.shape(reformer_output_grad[0]))

        #start rev_net backward
        f_ = self.reformer.model_layers.blocks[0].f
        g_ =  self.reformer.model_layers.blocks[0].g
        f_weights = f_.trainable_variables
        g_weights = g_.trainable_variables
        variables_names = [v.name for v in f_weights]
        tf.print(variables_names)

        
        y = self.reformer_output
        dy = reformer_output_grad[0]
        y = tf.concat([y, y], axis = -1) #revnet
        dy = tf.concat([dy, dy], axis = -1) #revnet
        tf.print(y)
        tf.print(tf.shape(y))
        tf.print(dy)
        tf.print(tf.shape(dy))


        dy1, dy2 = tf.split(dy, num_or_size_splits=2, axis=-1)#split last dimension
        del dy
        tf.print("+++++++++++++++")
        tf.print(tf.shape(dy1))
        tf.print(tf.shape(dy2))
        
   
        y1, y2 = tf.split(y, num_or_size_splits=2, axis=-1)  
        del y 


        z1_stop =  tf.stop_gradient(y1)




        gz1 = g_(z1_stop, training=training)
        
        x2 = y2 - gz1
        x2_stop = tf.stop_gradient(x2)

        fx2 = f_(x2_stop, training=training)
        x1 = y1 - fx2
        x1_stop = tf.stop_gradient(x1)

        #forward
        z1 = x1_stop + fx2
        y2 = x2_stop + gz1
        y1 = z1 

        grads_combined_1 = tf.gradients(
            y2, [z1_stop] + g_weights, grad_ys=dy2)

        tf.print(grads_combined_1[0])
        tf.print(tf.shape(grads_combined_1[0])) 

        dz1 = dy1 + grads_combined_1[0]
        dg = grads_combined_1[1:]
        dx1 = dz1

        
        
        tf.print("flag1++++++++++++")
        tf.print(x2_stop)
        tf.print(y1)
        tf.print("flag1++++++++++++")
        #xx = tf.stop_gradient(f_weights)
        #tf.print(f_weights)
        grads_combined_2 = tf.gradients(
            y1, [x2_stop] + f_weights, grad_ys=dz1)
        tf.print(grads_combined_2)
        #tf.print(tf.shape(grads_combined_2))

        """
        dx2 = dy2 + grads_combined_2[0]

        df = grads_combined_2[1:]


            

        grads = df + dg
        vars_ = f_weights + g_weights
        x = tf.concat([x1, x2], axis=-1)
        dx = tf.concat([dx1, dx2], axis=-1)
        
        #return x, dx, grads, vars_
        """
        """
        y, dy, grads, vars_ = block.backward_grads_and_vars(
            y, dy, training=training)
        grads_all += grads
        vars_all += vars_
        """
        #tf.print(tf.shape(dense_grad))
        #del tape
        #tf.print("(((((((((((((((")
        #dy, grads_all, vars_all=self.reformer.model_layers.backward_grads_and_vars(self.reformer_output,reformer_output_grad[0])
        #tf.print(dy.shape)
        #tf.print(grads_all.shape)




