import tensorflow as tf
import numpy as np
import os
import psutil
from w2v_data import get_data
import sys
import w2v_config as cfg


vocab_size = cfg.vocab_size
embed_size = cfg.embed_size
num_cwords = cfg.num_cwords
num_negsamples = cfg.num_negsamples
batch_size = cfg.batch_size
num_epochs = cfg.num_epochs
num_gpus = cfg.num_gpus

params = (vocab_size,embed_size,num_cwords,
          num_negsamples,batch_size,num_epochs,
          num_gpus)


def get_var_oncpu(varname,initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(varname,
      initializer=initializer,dtype=dtype)

  return var

def get_all_vars():
  embedding_matrix = get_var_oncpu(
    varname = "embedding_matrix",
    initializer = tf.random_uniform(
      shape = [vocab_size, embed_size], 
      minval = -1.0, maxval = 1.0))

  nce_weights = get_var_oncpu(
    varname = "nce_weights",
    initializer = tf.truncated_normal(
      shape = [vocab_size, embed_size], 
      stddev = 1.0 / np.sqrt(embed_size)))

  nce_biases = get_var_oncpu(
    varname = "nce_biases",
    initializer = tf.zeros(
      shape = [vocab_size]))

  return embedding_matrix, nce_weights, nce_biases


# given target predict context
def get_loss_op(target_id, context_id):

  embedding_matrix, nce_weights, nce_biases = get_all_vars() 

  # use embedding matrix to embed target index
  embedded_target = tf.nn.embedding_lookup(
    params = embedding_matrix, 
    ids = target_id)

  # why do I need to embed target word 
  # but I pass context index to loss?
  # is it because I am classifying that target vector
  # to the class that corresponds to context index?
  loss_per_ex = tf.nn.nce_loss(
      weights = nce_weights,
      biases = nce_biases,
      labels = context_id,
      inputs = embedded_target, 
      num_sampled = num_negsamples, 
      num_classes = vocab_size,
      name = 'loss_per_example')

  loss = tf.reduce_mean(loss_per_ex, 
      name = 'bmean_loss_op')

  return loss


def get_dsitr(tph,cph):

  tfds = tf.contrib.data.Dataset.from_tensor_slices((tph, cph))
  tfds = tfds.repeat(num_epochs)
  tfds = tfds.shuffle(100000)
  tfds = tfds.batch(batch_size)
  dsitr = tfds.make_initializable_iterator()
  
  return dsitr



def train_w2v(corpus_fpath):

  print('-- TRAINING ON', corpus_fpath)

  w2vgraph = tf.Graph()

  ## train ##
  pconfig = tf.ConfigProto(log_device_placement=False,
                          allow_soft_placement=True)
  with tf.Session(graph=w2vgraph,config=pconfig) as sess:
    
    # dataset iterator
    with tf.device('/cpu:0'):
      data_dict, i2w_dict = get_data(
        corpus_fpath, vocab_size, num_cwords)
      tph = tf.placeholder(tf.int64,
          shape=[len(data_dict['target'])],
          name='target_placeholder')
      cph = tf.placeholder(tf.int64,
          shape=[len(data_dict['context'])],
          name='context_placeholder')
      feed_dict = {tph:data_dict['target'],
                    cph:data_dict['context']}
      dsitr = get_dsitr(tph,cph)
      sess.run(dsitr.initializer,feed_dict)
      
    # assembling towers
    applygrads_list = []
    scope = tf.get_variable_scope()
    for g in range(num_gpus):
      with tf.device('/gpu:%i'%g), tf.name_scope('Tower%i'%g):
        # get next sample
        target_b, context_b = dsitr.get_next()
        context_b = tf.expand_dims(context_b,axis=1)
        # setup loss and optimizer
        loss_op = get_loss_op(target_b,context_b)
        # compute batch gradients
        optimizer_op = tf.train.GradientDescentOptimizer(
          learning_rate=0.05, name='optimizer_op')
        grads_and_vars = optimizer_op.compute_gradients(loss_op)
        applygrads_op = optimizer_op.apply_gradients(grads_and_vars)
        applygrads_list.append(applygrads_op)
        scope.reuse_variables()

    applygrads_group = tf.group(*applygrads_list)
    tf.global_variables_initializer().run()

    # train loop
    while True:
      try:
        opt_return, batch_loss = sess.run([applygrads_group, loss_op])
        print(batch_loss)
      except tf.errors.OutOfRangeError:
        print('ended iter')
        break

    # store normalized embeddings into array
    with tf.variable_scope(tf.get_variable_scope(),reuse=True):
      embedding_matrix = tf.get_variable('embedding_matrix')
      norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_matrix), 1, keep_dims=True))
      normalize_embed_matrix = embedding_matrix / norm
      normal_embed_matrix = normalize_embed_matrix.eval()

    embed_dict = make_embed_dictionary(i2w_dict,normal_embed_matrix)

  return embed_dict


def make_embed_dictionary(i2w_dict,embedding_matrix):
  """ make a dictionary of vectors indexed by word
      called in .train() method """
  embed_dict = dict()
  for index,word in i2w_dict.items():
    embed_dict[word] = embedding_matrix[index]
  return embed_dict


if __name__ == "__main__":

  # corpus_fpath = sys.argv[1]
  # results_dir = sys.argv[2]
  corpus_fpath = "/Users/abeukers/wd/w2v/data/corpus_partitions/4_k10/FOX-1of10.txt"
  results_dir = "/Users/abeukers/wd/w2v/results/troubleshoot"

  param_str = "%ivocab_%iembed_%icwords_%ineg_%ibatch_%iepo_%igpu" % params
  with open(results_dir+"/0params.txt", 'w') as file:
    file.write(param_str)
  print('\n parameters: %s \n' % param_str)

  # train
  embed_dict = train_w2v(corpus_fpath)

  # save embed dict
  np.save(results_dir + '/%s_embed_dict' % 
    corpus_fpath.split('/')[-1].split('.')[0], embed_dict)

  













