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
from .TFattention import TFSelfAttention, TFFeedForward,MultiHeadAttention
from .TFutils import cache_fn, Chunk, WithNorm
from .blocks import ReversibleBlock, ReversibleSequence

class TFReformer(tf.keras.Model):
    def __init__(self, emb, depth, max_seq_len, heads = 8, bucket_size = 64, 
                        n_hashes = 8, ff_chunks = 100, attn_chunks = None, 
                        causal = False, weight_tie = False, lsh_dropout = 0., 
                        lsh_attend_across_buckets = False, lsh_allow_duplicate_attention = True, 
                        random_rotations_per_head = False, twin_attention = False, 
                        use_scale_norm = False, use_full_attn = False):
        super().__init__()
        self.emb = emb
        self.depth = depth

        get_full_attn = lambda: TFSelfAttention(emb, heads, causal = causal)
        get_lsh_attn = lambda: TFLSHSelfAttention(emb, heads, bucket_size, n_hashes, 
                                                    causal = causal, dropout = lsh_dropout, attn_chunks = attn_chunks, 
                                                    allow_duplicate_attention = lsh_allow_duplicate_attention, attend_across_buckets = lsh_attend_across_buckets, 
                                                    random_rotations_per_head = random_rotations_per_head)

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

    def call(self, x,training=True):
        
        
        #tf.print(tf.equal(tf.stack(tf.reduce_sum(tf.split(x, 2, axis=-1), axis=0)),tf.reduce_sum(tf.split(x, 2, axis=-1), axis=0)))
        x = self.model_layers(x,training=training)
        return x

class TFReformerLM(tf.keras.Model):
    def __init__(self, num_tokens, emb, depth, max_seq_len, heads = 8, bucket_size = 64, n_hashes = 8, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, use_full_attn = False):
        super().__init__()
        self.token_emb = Embedding(num_tokens, emb)
        self.pos_emb = Embedding(max_seq_len, emb)
        self.reformer = TFReformer(emb, depth, max_seq_len, heads = heads,lsh_attend_across_buckets = False, 
                                    bucket_size = bucket_size, n_hashes = n_hashes, ff_chunks = ff_chunks, 
                                    attn_chunks = attn_chunks, causal = causal, weight_tie = weight_tie, 
                                    lsh_dropout = lsh_dropout, random_rotations_per_head = random_rotations_per_head, 
                                    twin_attention = twin_attention, use_scale_norm = use_scale_norm, 
                                    use_full_attn = use_full_attn)
        self.lastlayernorm = LayerNormalization()
        self.to_logits = Dense(num_tokens)
        self.reformer_output=None
        self.optimizer = None

    def set_optimizer(self,optimizer):
        self.optimizer = optimizer

    def create_checkpoint_manager(self, checkpoint_path, max_to_keep=5, load_model=True):
        with tf.name_scope('checkpoint_manager'):
            ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
            self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

            if load_model:  # If want to load trained weights
                ckpt.restore(self.ckpt_manager.latest_checkpoint)
                print('Latest checkpoint restored...............')
            else:
                print("Initializing model from scratch..........")
    def load_model(self, filepath):
        ckpt = tf.train.Checkpoint(model=self)
        ckpt_manager = tf.train.CheckpointManager(ckpt, filepath,max_to_keep=1)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Model Restored..........................")


    def before_reformer(self, inputs):
        
        inputs = self.token_emb(inputs)+ self.pos_emb(tf.range(inputs.shape[1]))
        inputs = tf.concat([inputs, inputs], axis = -1) #revnet
        return inputs

    def after_reformer(self, inputs):
        
        inputs = tf.reduce_sum(tf.split(inputs, 2, axis=-1), axis=0) #merge the split input at here 
        reformer_outputs = self.lastlayernorm(inputs)
        logits = self.to_logits(reformer_outputs)
        return logits


    def call(self, inputs,training=True):
        embedded_inputs = self.before_reformer(inputs)
        #tf.print(embedded_inputs)

        reformer_output = self.reformer(embedded_inputs,training=training)
        logits = self.after_reformer(reformer_output)
        return logits


 
    def train_step(self,inputs,targets,loss_object,loss_metric,mirrored_strategy=None, training=True,distributed = False ):
        if distributed :

            def step_fn(inputs,targets,loss_object,loss_metric,training=True):

                loss, grads_all, vars_all, cross_entropy = self.backward_grads_and_vars(inputs,targets,loss_object,training=training)

                self.optimizer.apply_gradients(zip(grads_all, vars_all))
                loss_metric(loss)   

                return cross_entropy


            per_example_losses = mirrored_strategy.experimental_run_v2(
                step_fn, args=(inputs,targets,loss_object,loss_metric,True,))

            loss = mirrored_strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0) 
            
        else:             
            loss, grads_all, vars_all, _ = self.backward_grads_and_vars(inputs,targets,loss_object,training=training)

            self.optimizer.apply_gradients(zip(grads_all, vars_all))
            loss_metric(loss)

        return loss

    @tf.function 
    def eval_step(self,inputs,targets,loss_object,loss_metric,mirrored_strategy=None, training=False,distributed = False ):
        if distributed :

            def step_fn(inputs,targets,loss_object,loss_metric,training=True):

                loss, grads_all, vars_all, cross_entropy = self.backward_grads_and_vars(inputs,targets,loss_object,training=training)

                loss_metric(loss)   

                return cross_entropy


            per_example_losses = mirrored_strategy.experimental_run_v2(
                step_fn, args=(inputs,targets,loss_object,loss_metric,True,))

            loss = mirrored_strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0) 
            
        else:             
            loss, grads_all, vars_all, _ = self.backward_grads_and_vars(inputs,targets,loss_object,training=training)

            loss_metric(loss)
        return loss   

    def backward_grads_and_vars(self,inputs,targets,loss_object,training=True):
        total_grads_all = []
        total_vars_all = []
        def get_loss(real, pred, loss_object):
            with tf.name_scope("loss_layer"):
                mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 0)),tf.float32)
                loss_ = loss_object(real, pred) #loss : (batch , seq_len), real: (batch , seq_len), pred: (batch , seq_len, voc)
                loss_ = loss_ * mask 
                loss_ = tf.reduce_sum(loss_, axis=1)
                sequence_avg_loss = loss_ / tf.reduce_sum(mask, axis=1) #batch 
                
                return sequence_avg_loss

        with tf.GradientTape() as tape_0:
            inputs = tf.identity(inputs)
            embedded_inputs = self.before_reformer(inputs)


        reformer_outputs = self.reformer(embedded_inputs,training=training)
        #tf.print(reformer_outputs.shape)


        #for gradient of logits
        with tf.GradientTape() as tape_1:
            reformer_outputs = tf.identity(reformer_outputs)
            tape_1.watch(reformer_outputs)
            logits = self.after_reformer(reformer_outputs)
        

        #tf.print(y_hat[0][0])
            cross_entropy = get_loss(targets, logits ,loss_object)
            loss = tf.reduce_mean(cross_entropy)
        

        #tf.print(self.to_logits.trainable_variables)
        #tf.print(tape_1.watched_variables())
        #update output layer
        dense_variable_list = list(tape_1.watched_variables())
        dense_grad_result = tape_1.gradient(loss, dense_variable_list + [reformer_outputs])
        dense_grad, reformer_outputs_grad = dense_grad_result[:-1], dense_grad_result[-1]



        total_grads_all.extend(dense_grad)
        total_vars_all.extend(tape_1.watched_variables())



        
        y, dy, grads_all, vars_all = self.reformer.model_layers.backward_grads_and_vars(reformer_outputs,reformer_outputs_grad)

        total_grads_all.extend(grads_all)
        total_vars_all.extend(vars_all)


        del tape_1

        #update word embeddings
        word_embedding_variable_list = list(tape_0.watched_variables())
        we_grad_result = tape_0.gradient(embedded_inputs, word_embedding_variable_list,dy )
        #tf.print(tf.reduce_sum(we_grad_result))

        del tape_0


        total_grads_all.extend(we_grad_result)
        total_vars_all.extend(word_embedding_variable_list)
        total_grads_all = [tf.clip_by_norm(g, 0.5)
             for g in total_grads_all]

        return loss, total_grads_all, total_vars_all, cross_entropy
     