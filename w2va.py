import numpy as np
from glob import glob
import psutil
import os


def get_common_vocab(embed_dicts):
  """ creates a set common_vocab with words common to all corpora
      must be trained"""
  # NB! set operation is mixing up word ordering

  # take intersection of every vocabulary in w2v_dict
  vocabs = [d.keys() for d in embed_dicts.values()]
  common_vocab = set.intersection(*map(set,vocabs))
  print('\n there are %i common words' % len(common_vocab))    
  return common_vocab


def get_wordRSM(embed_dict,vocab):
  """ for a given embed_dict """

  embeds = np.array([embed_dict[w] for w in vocab])
  wordRSM = np.corrcoef(embeds)
  print('wrsm shape is',wordRSM.shape)

  # wordRSM = np.matmul(embeds,embeds.transpose())

  return wordRSM
    

def get_corpRSM(embed_dicts,vocab):
  """ stores the corpus_similarity matrix, containing the similarity between corpora
       implementation: vectorize the word_similairty_within matrices for every trained model
        compute X X'   
      """
  wordRSMs=[]
  print('corpora in this RSM',embed_dicts.keys())
  for i,embed_dict in enumerate(embed_dicts.values()):
    print("wordRSM: ", i)

    wordRSM = get_wordRSM(embed_dict,vocab).reshape(-1,1)
    wordRSMs.append(wordRSM)
    # print(wordRSMs.shape)
    # wordRSM = wordRSM/np.linalg.norm(wordRSM)
    print(wordRSM.shape)

      
  wordRSMs = np.array(wordRSMs).squeeze()
  print('wordRSMs shape is',wordRSMs.shape)

  corpRSM = np.corrcoef(wordRSMs)
  print('corprsm shape is', corpRSM.shape)
  return corpRSM























