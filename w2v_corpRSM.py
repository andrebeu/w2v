import os
from os.path import join as opj
from glob import glob
import numpy as np
import w2va


results_dir = "results/14"

print(os.listdir(results_dir)[1:-1])


# load embed_dicts into dict indexed by corp name
embed_dicts = {}
R = os.listdir(results_dir)
for r in R:
	print("loaded",r)
	embed_dicts[r.split('_')[0]] = np.load(opj(results_dir,r)).item()

common_vocab = w2va.get_common_vocab(embed_dicts)
corpRSM = w2va.get_corpRSM(embed_dicts,common_vocab)

np.save(results_dir+'/corpRSM', corpRSM)







