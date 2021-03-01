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
import time

class ReversibleSequence(tf.keras.Model):
    """Single reversible block containing several `_Residual` blocks.
    Each `_Residual` block in turn contains two _ResidualInner blocks,
    corresponding to the `F`/`G` functions in the paper.

    This is based on PyTorch's RevTorch - ReversibleSequence
    """

    def __init__(self,
                blocks):
        """Initialize RevBlock.
        Args:
        n_res: number of residual blocks
        filters: list/tuple of integers for output filter sizes of each residual
        strides: length 2 list/tuple of integers for height and width strides
        input_shape: length 3 list/tuple of integers
        batch_norm_first: whether to apply activation and batch norm before conv
        data_format: tensor data format, "NCHW"/"NHWC"
        bottleneck: use bottleneck residual if True
        fused: use fused batch normalization if True
        dtype: float16, float32, or float64
        """
        super(ReversibleSequence, self).__init__()
        self.blocks = blocks

        self.block_kwargs = {'f_args': {}, 'g_args': {}}


    def call(self, h, training=True):
        """Apply reversible block to inputs."""
        for layer_i,block in enumerate(self.blocks):
            h = block(h, layer_i, **self.block_kwargs,  training=training)
        return h

    def backward_grads_and_vars(self, y, dy, training=True):
        """Apply reversible block backward to outputs."""
        
        grads_all = []
        vars_all = []
        #tf.print(y.shape)
        #tf.print(dy.shape)


        for i in reversed(range(len(self.blocks))):
            #print(i)
            block = self.blocks[i]
            layer_i = i
            y, dy, grads, vars_ = block.backward_grads_and_vars(
                y, dy, layer_i, **self.block_kwargs, training=training)
            grads_all += grads
            vars_all += vars_


        return y,dy, grads_all, vars_all



class ReversibleBlock(tf.keras.Model):
    """Single residual block contained in a _RevBlock. Each `_Residual` object has
    two _ResidualInner objects, corresponding to the `F` and `G` functions in the
    paper. This version takes in the F and G block directly, instead of constructing them. 

    This implementation is based on PyTorch's RevTorch - ReversibleBlock
    Args:
        f_block: The first residual block
        g_block: the second residual block
        split_along_axis: axis for splitting, defaults to 1
    """

    def __init__(self,
                f_block,
                g_block,
                split_along_axis=-1):
        super(ReversibleBlock, self).__init__()

        self.axis = split_along_axis        
        self.f = f_block
        self.g = g_block
        self.f_seed = None
        self.g_seed = None

    def call(self, x, layer_i,  f_args = {}, g_args = {}, training=True):
        """Apply residual block to inputs."""

        #set the dropouted weights same in revesrible proccess
        self.f_seed = time.time()
        self.g_seed = time.time()*10.0


        x1, x2 = tf.split(x, num_or_size_splits=2, axis=self.axis)


        f_args['_reverse'] = g_args['_reverse'] = False
        f_args['_layer_i'] = g_args['_layer_i'] = layer_i
        f_args['_seed'] =  self.f_seed
        g_args['_seed'] =  self.g_seed

        f_x2 = self.f(x2, **f_args)
        y1 = f_x2 + x1
        g_y1 = self.g(y1, **g_args)
        y2 = g_y1 + x2


        """
        self.x2 = x2
        self.x1 = x1
        self.f_x2 = f_x2
        """

        return tf.concat([y1, y2], axis=self.axis)


    def backward_grads_and_vars(self, y, dy, layer_i, f_args = {}, g_args = {}, training=True):

        #Manually compute backward gradients given input and output grads.
        dy1, dy2 = tf.split(dy, num_or_size_splits=2, axis=self.axis)#split last dimension

        del dy
        y1, y2 = tf.split(y, num_or_size_splits=2, axis=self.axis)  
        
        del y

        f_args['_reverse'] = g_args['_reverse'] = True
        f_args['_layer_i'] = g_args['_layer_i'] = layer_i
        f_args['_seed'] =  self.f_seed
        g_args['_seed'] =  self.g_seed


        #start rev_net backward
        f_ = self.f
        g_ =  self.g
        f_weights = f_.trainable_variables
        g_weights = g_.trainable_variables
 
        z1= y1

        with tf.GradientTape() as tape_2:
            
            tape_2.watch(z1)
           
            g_z1 = g_(z1, **g_args )
        

        grad_result = tape_2.gradient(g_z1,g_weights+[z1],dy2)

        del tape_2

        dg_weights, dz1 = grad_result[:-1], grad_result[-1]



        x2 = y2 - g_z1
        del y2, g_z1
        """
        tf.print("=="*100)
        tf.print(tf.reduce_sum(x2-self.x2))
        """
        dx1 = dy1 + dz1
        del dy1, dz1
        
        with tf.GradientTape() as tape_3:
            tape_3.watch(x2)
            f_x2 = f_(x2,  **f_args)
    
        grad_result = tape_3.gradient(f_x2, f_weights+[x2],dx1)
        df_weights, dx2 = grad_result[:-1], grad_result[-1]
       
        x1 = y1 - f_x2
        """
        tf.print("@"*100)
        tf.print(tf.reduce_sum(x1-self.x1))


        tf.print("$"*100)
        tf.print(tf.reduce_sum(f_x2-self.f_x2))
        """
        del y1, f_x2




        dx2 = dx2 + dy2
        del  dy2
        grads_ = df_weights + dg_weights
        vars_ = f_weights+ g_weights

        x = tf.concat([x1, x2], axis=self.axis)
        dx = tf.concat([tf.stop_gradient(dx1), tf.stop_gradient(dx2)], axis=self.axis)
        
        return x, dx, grads_, vars_
