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
SELF_ATTN_INF_NEG = -5e4
LOOK_AHEAD_ATTN_INF_NEG = -1e38
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from .utils import cache_method_decorator,sort_key_val, batched_index_select,merge_dims, make_unit_length, process_inputs_chunk,get_padding
class TFLSHAttention(tf.keras.Model):
    def __init__( self,
                  dropout = 0.,
                  bucket_size = 64,
                  n_hashes = 8,
                  causal = False,
                  allow_duplicate_attention = True,
                  attend_across_buckets = True,
                  rehash_each_round = True,
                  drop_for_hash_rate = 0.0,
                  random_rotations_per_head = False,
                  training=True):
        super(TFLSHAttention, self).__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = dropout
        
        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head
        self.seed_ = None
        self.buckets = None
        self.training = training
        # cache buckets for reversible network, reported by authors to make Reformer work at depth
        self._cache = {}


    @cache_method_decorator('_cache', 'buckets', reexecute=True)    
    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0
        
        rot_size = n_buckets #rotation몇번할꺼냐 혹은 몇개의 bucket 으로 나눌거야 

        rotations_shape = (
            1 ,
            vecs.shape[-1],
            self.n_hashes ,
            rot_size // 2)

        #rotations_shape[0] == 1 일때 해당차원 batch size로 뿔린다
        tf.random.set_seed(self.seed_)

        random_rotations = tf.random.normal(rotations_shape, dtype=tf.dtypes.float32)
        random_rotations = tf.tile(random_rotations, (batch_size,1,1,1))


        rotated_vecs = tf.einsum('btf,bfhi->bhti', vecs, random_rotations)
        rotated_vecs = tf.concat([rotated_vecs, -rotated_vecs], axis=-1)

        buckets = tf.math.argmax(rotated_vecs, axis=-1) #여기서 buckets는 bucket number 들의 matrix라는 뜻 

        #tf.print(buckets[0][0])
        # buckets is now (batch, self.n_hashes, seqlen). Next we add offsets so that
        # bucket numbers from different hashing rounds don't overlap.
        offsets = tf.range(self.n_hashes) #[0..self.n_hashes-1]
        offsets = tf.reshape(offsets * n_buckets, (1, -1, 1))#[0..n_buckets*(self.n_hashes-1)]    argmax값들이 다른 Hash round와 중복하지 않기위해 값을 곱하였다.  ex) [0,n_buckets,2*n_buckets,..]
        
        offsets = tf.cast(offsets, tf.int64)
        buckets = tf.reshape(buckets + offsets, (batch_size, -1,)) # batch size 남기고 collapse

        """
        if self.buckets != None:

            buckets = self.buckets
            self.buckets=None
        else:
            self.buckets=buckets
        """

        return buckets#(batch, self.seqlen*n_hashes)



    def call(self, qk, v,**kwargs):

        is_reverse = kwargs.pop('_reverse', False)
        layer_i = kwargs.pop('_layer_i', None)
        _seed = kwargs.pop('_seed', None)


        self.seed_ = _seed
        tf.random.set_seed(self.seed_)

        batch_size, seqlen, dim = qk.shape
        device = qk.device

        n_buckets = seqlen // self.bucket_size #몇개의 bucket으로 나눌것인가 
        n_bins = n_buckets



        buckets = self.hash_vectors(n_buckets, qk, key_namespace=layer_i, fetch=is_reverse, set_cache=self.training)
        #buckets = self.hash_vectors(n_buckets, qk)
        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == seqlen * self.n_hashes 
        ticker = tf.expand_dims(tf.range(seqlen * self.n_hashes ), axis=0)
        buckets_and_t = seqlen * buckets + tf.cast((ticker % seqlen), tf.int64)#to sort q by bucket number and by seq length

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, axis=-1)

        _, undo_sort = sort_key_val(sticker, ticker, axis=-1)
        del ticker


        st = (sticker % seqlen) #index참조하기위해 다시 원상복구  seqlen * buckets 부분 필요없어짐 

        #bucket 안에emb값이 바슷한것끼리 묶임 , n_hash는 분리됨아직 
        sqk = batched_index_select(qk, st) #batch, seqlen*n_hashes,emb_dim 
        
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        bq_t = bkv_t = tf.reshape(st, (batch_size, self.n_hashes * n_bins, -1))#batch, n_bins*n_hashes,bucket_size
        bqk = tf.reshape(sqk, (batch_size, self.n_hashes * n_bins, -1, sqk.shape[-1]))#batch, n_bins*n_hashes,bucket_size,emb_dim
        bv = tf.reshape(sv, (batch_size, self.n_hashes * n_bins, -1, sv.shape[-1]))
        bq_buckets = bkv_buckets = tf.reshape(sbuckets_and_t // seqlen, (batch_size, self.n_hashes * n_bins, -1))#batch, n_bins*n_hashes, bucket size #n_hashs offset 한건 살아있음 
        
        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = make_unit_length(bqk)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = tf.concat([x[:, -1:, ...], x[:, :-1, ...]], axis=1)
            return tf.concat([x, x_extra], axis=2)
   
        bk = look_one_back(bk)#batch, n_bins*n_hashes, chunk(2*bucket_size), emb_dim
      
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)
        bkv_buckets = look_one_back(bkv_buckets)

        # Dot-product attention.
        dots = tf.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)#batch, n_bins*n_hashes,bucket_size,2*bucket_size # only bucket_sizex2*bucket_size!!!
        #print(dots.shape)
        #assume the dataset has no padded and packed fully the entire seq_len 

        # Causal masking
        #tf.print("asfd"*100)
        #tf.print(self.causal)
        
        
        if self.causal:
   
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :] #index 관한 마스크 >t 인것들 마스킹 

            #tf.print(mask)
            dots = tf.math.multiply(dots, (1-tf.cast(mask, tf.float32))) + (tf.cast(mask, tf.float32)) * (LOOK_AHEAD_ATTN_INF_NEG )
            del mask
        
        
        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :] # 자기자신인것 index마스킹 
        #tf.print(self_mask)
        dots = tf.math.multiply(dots, (1-tf.cast(self_mask, tf.float32))) + (tf.cast(self_mask, tf.float32)) * (SELF_ATTN_INF_NEG )
        del self_mask
        # Mask out attention to other hash buckets.
        
        if not self._attend_across_buckets:
            #tf.print("###"*100)
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :] #내용중 자기 랑 다른 Hash인것들 2*chunk size 단위로 마스킹 
            #tf.print(bucket_mask)
          
            dots = tf.math.multiply(dots, (1-tf.cast(bucket_mask, tf.float32))) + (tf.cast(bucket_mask, tf.float32)) * (LOOK_AHEAD_ATTN_INF_NEG )
            
            #tf.print(dots)
            #tf.print(bucket_mask)
        


        dots_logsumexp = tf.math.reduce_logsumexp(dots, axis=-1, keepdims=True) #2*bucket_size 에 대한 partition function 

        dots = tf.exp(dots - dots_logsumexp)#softmax 
        

        dots = tf.nn.dropout(dots, rate=0.1)
        
        bo = tf.einsum('buij,buje->buie', dots, bv)#batch, n_bins*n_hashes, bucket_size, emb_size
        so = tf.reshape(bo, (batch_size, -1, bo.shape[-1]))#batch, seq_len * n_hashes, emb_size
        
        slogits = tf.reshape(dots_logsumexp, (batch_size, -1,)) #batch, seq_len * n_hashes
        #tf.print("="*100)
        #tf.print(slogits)
        class UnsortLogits(tf.keras.layers.Layer):
            def __init__(self):
                super(UnsortLogits, self).__init__()
            
            def call(self, so, slogits):
                #so, slogits = tf.stop_gradient(so), tf.stop_gradient(slogits)

                o = batched_index_select(so, undo_sort)
                _, logits = sort_key_val(sticker, slogits, axis=-1)
                return o, logits

            
        unsortlogits = UnsortLogits()
        o, logits = unsortlogits(so, slogits) ##batch, seq_len * n_hashes 에서 마지막 차원 의 logit


        if self.n_hashes == 1:
            out = o
        else:#collapse hashes
            o = tf.reshape(o, (batch_size, self.n_hashes, seqlen, o.shape[-1]))

            logits = tf.reshape(logits, (batch_size, self.n_hashes, seqlen, 1))


            logits_logsumexp =  tf.math.reduce_logsumexp(logits, axis=1, keepdims=True)#(batch_size, 1, seqlen, 1)


            probs = tf.exp(logits - logits_logsumexp)# softmax to do weight sum on n_hashes dim ###(batch_size, self.n_hashes, seqlen, 1)

            out = tf.reduce_sum(o * probs, axis=1)
 
        
        assert out.shape == v.shape
        return out

        
class TFLSHSelfAttention(tf.keras.Model):
    def __init__(self, d_model, num_heads = 8, bucket_size = 64, n_hashes = 8, causal = False, attn_chunks = None, random_rotations_per_head = False, attend_across_buckets = False, allow_duplicate_attention = True, **kwargs):
        super(TFLSHSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0 , 'dimensions must be divisible by number of heads'
        self.depth = d_model // self.num_heads
        self.qk_dense = Dense(units=d_model, use_bias=False)
        
        self.value_dense = Dense(units=d_model, use_bias=False)
        self.dense = Dense(units=d_model)

        self.layer_num= None 
        self.seed = None 

        self.bucket_size = bucket_size
        self.lsh_attn = TFLSHAttention(bucket_size=bucket_size, n_hashes=n_hashes, causal=causal, random_rotations_per_head=random_rotations_per_head, attend_across_buckets = attend_across_buckets,  allow_duplicate_attention = allow_duplicate_attention, **kwargs)

    def call(self, inputs, **kwargs):
        b, t, e, h = *inputs.shape, self.num_heads
        assert t % self.bucket_size == 0, f'Sequence length needs to be divisible by target bucket size - {self.bucket_size}'
        assert e == self.d_model
        d = e // h

        qk = self.qk_dense(inputs)
        v = self.value_dense(inputs)

        def merge_heads(v):
            return tf.reshape(tf.transpose(v, perm=[0, 2, 1, 3]), (b , -1,  e)) 

        def split_heads(v):
            return tf.transpose(tf.reshape(v, (b, -1, h, d)), perm=[0, 2, 1, 3])

        
        qk = merge_dims(0,1,split_heads(qk))
        v = merge_dims(0,1,split_heads(v))


        outputs = self.lsh_attn(qk, v, **kwargs)

        outputs = tf.reshape(outputs, (b,h,t,d))

        out = merge_heads(outputs)
#        out = self.to_out(out)
#        return self.post_attn_dropout(out)
        return self.dense(out)
