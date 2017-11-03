import os
from os.path import join as opj
import collections
import json



## -- MAKING CORPUS PARTITIONS

output_dir = opj('data','corpus_partitions')
preprocessed_dir = opj('data','preprocessed_corpora')


## partitioning transcripts from each channel into k partitions

def random_partition_list(L,k):
  from numpy import random
  """randomize and partition a list 
    returns a list of k lists"""
  random.shuffle(L)
  n = int(len(L) / k)
  partitions = [L[i:i+n] for i in range(0,n*k,n)]
  return partitions

def get_paths_dict(k):
  """ returns dict containing k lists of paths for each channel """

  D = dict()
  all_paths = []
  # for each channel
  for ch in ['FOX','MSNBC']:
    ch_paths = [opj(preprocessed_dir,ch,f) for f in 
      os.listdir(opj(preprocessed_dir,ch))]
    D[ch] = random_partition_list(ch_paths,k)
    all_paths.extend(ch_paths)

  return D


## writing transcripts to larger corpus

def make_one_corpus(fpath_list):
  """ input a list of paths to episode transcripts
    outputs a corpus concatenating all transcripts"""
  corp = ""
  for fpath in fpath_list:
    with open(fpath) as file:
      transcript = file.read()
    corp += transcript
    # corp += END_TOKEN
  return corp


## main function

def make_corpus(k=2):
  """ k corpora per channel """

  paths_dict = get_paths_dict(k)
  # make new folder for corpus creation
  corp_folder = opj(output_dir,
    str(len(os.listdir(output_dir)))+"_k%i"%k)
  os.mkdir(corp_folder)
  
  for ch,L in paths_dict.items():
    wcount = dict()
    for i,fpath_list in enumerate(L):
      corp = make_one_corpus(fpath_list)
      corp_fpath = opj(corp_folder,"%s-%iof%i.txt"%(ch,i+1,k))
      with open(corp_fpath,'w') as file:
        file.write(corp)
      wcount[i] = len(corp.split())

    with open(opj(corp_folder,"0wcount-%s.txt"%ch),'w') as file:
      file.write(json.dumps(wcount))
    with open(opj(corp_folder,"0transcripts-%s.txt"%ch),'w') as file:
      file.write(json.dumps(L))




## -- GETTING DATA FOR TRAINING


def make_corpi(corpus, vocab_size):
  """takes a word corpus, assigns an id to each word
  returns a string of ids corresponding to the words
  and an index2word dict"""

  corpus = corpus.split()

  # frequency of occurence of each word
  allwords_freq = collections.Counter(corpus)
  commonwords_freq = [['UNK',-1]]
  commonwords_freq.extend(allwords_freq.most_common(vocab_size -1))

  # assign index to each word
  word_index = dict()
  for w,_ in commonwords_freq:
    word_index[w] = len(word_index)
    # print(word_index)

  corpi = list()
  for w in corpus:
    if w in word_index:
      wi = word_index[w]
    else: 
      wi = word_index['UNK']

    corpi.append(wi)

  index2word = dict(zip(word_index.values(),word_index.keys()))

  return corpi, index2word



def get_data_dict(corpi,num_cwords):
  assert num_cwords % 2 == 0, '--> num_cwords invalid'
  """ takes an indecized corpus (corpi), returns dataset
  dataset consists of a dict of lists 
  {'target':[t1,t2...], 'context':[c1,c2...]} """

  # window to slide over
  window_size = num_cwords + 1
  window = collections.deque(maxlen=window_size)

  # initialize window
  for i in range(window_size):    
    window.append(corpi[i])

  # index for context and target words in window
  t_index = int(num_cwords/2)
  c_index = [i for i in range(num_cwords+1) if i != t_index]

  target = []
  context = []
  for w in corpi:
    for c_i in c_index:
      target.append(window[t_index])
      context.append(window[c_i]) 
    # slide window
    window.append(w)

  return {'target':target, 'context':context}

## main function

def get_data(corpus_fpath,vocab_size,num_cwords):

  with open(corpus_fpath) as file:
    corpus = file.read()

  corpi, i2w_dict = make_corpi(corpus, vocab_size)

  return get_data_dict(corpi,num_cwords), i2w_dict

  










