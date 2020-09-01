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

    def call(self, h, training=True):
        """Apply reversible block to inputs."""
        for block in self.blocks:
            h = block(h, training=training)
        return h

    def backward_grads_and_vars(self, y, dy, training=True):
        """Apply reversible block backward to outputs."""

        grads_all = []
        vars_all = []
        #tf.print(y.shape)
        #tf.print(dy.shape)
        y = tf.concat([y, y], axis = -1) #revnet
        dy = tf.concat([dy, dy], axis = -1) #revnet
        for i in reversed(range(len(self.blocks))):

            block = self.blocks[i]
   
            y, dy, grads, vars_ = block.backward_grads_and_vars(
                y, dy, training=training)
            grads_all += grads
            vars_all += vars_

        y = tf.stack(tf.reduce_sum(tf.split(y, 2, axis=-1), axis=0))
        dy = tf.stack(tf.reduce_sum(tf.split(dy, 2, axis=-1), axis=0))

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
                split_along_axis=-11):
        super(ReversibleBlock, self).__init__()

        self.axis = split_along_axis        
        self.f = f_block
        self.g = g_block

    def call(self, x, training=True, concat=True):
        """Apply residual block to inputs."""

        x1, x2 = tf.split(x, num_or_size_splits=2, axis=self.axis)
        f_x2 = self.f(x2, training=training)
        y1 = f_x2 + x1
        g_y1 = self.g(y1, training=training)
        y2 = g_y1 + x2
        if not concat:  # For correct backward grads
            return y1, y2

        return tf.concat([y1, y2], axis=self.axis)


    def backward_grads_and_vars(self, y, dy, training=True):
        #Manually compute backward gradients given input and output grads.
        dy1, dy2 = tf.split(dy, num_or_size_splits=2, axis=self.axis)#split last dimension
   
        
        y1, y2 = tf.split(y, num_or_size_splits=2, axis=self.axis)  


        #start rev_net backward
        f_ = self.f
        g_ =  self.g
        f_weights = f_.trainable_variables
        g_weights = g_.trainable_variables
 
        


        z1_stop =  tf.stop_gradient(y1)
        gz1 = g_(z1_stop, training=training)
        
        x2 = y2 - gz1
        x2_stop = tf.stop_gradient(x2)

        fx2 = f_(x2_stop, training=training)
        x1 = y1 - fx2
        x1_stop = tf.stop_gradient(x1)

        #forward to get the gradient 
        z1 = x1_stop + fx2
        y2 = x2_stop + gz1
        y1 = z1 

        grads_combined_1 = tf.gradients(
            y2, [z1_stop] + g_weights, grad_ys=dy2)

        #tf.print(grads_combined_1[0])
        #tf.print(tf.shape(grads_combined_1[0])) 

        dz1 = dy1 + grads_combined_1[0]
        dg = grads_combined_1[1:]
        dx1 = dz1

   
        grads_combined_2 = tf.gradients(
            y1, [x2_stop] + f_weights, grad_ys=dz1)

        dx2 = dy2 + grads_combined_2[0]
        df = grads_combined_2[1:]
        
        #tf.print(dx2)
        #tf.print(df)

        grads = df + dg
        vars_ = f_weights+ g_weights

        x = tf.concat([x1, x2], axis=self.axis)
        dx = tf.concat([dx1, dx2], axis=self.axis)
        
        return x, dx, grads, vars_

