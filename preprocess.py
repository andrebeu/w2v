from os.path import join as opj
import os, json, re
import random

import enchant

""" preprocesses the text in each episode transcript
    writes preprocessed transcript 
"""

print('-------------START-------------')

# channel : MSNBC, FOX
# show: foxandfriends,hannity,...
# episode: hannity031, hannity032...

output_dir = opj('data','output','preprocessed')
raw_dir = opj('data','raw_data')

def preprocess(episode_fpath):

    """ read, process, write episode transcript"""  
    episode_fname = episode_fpath.split('/')[-1]
    print('processing',episode_fname)

    # read episode_transcript
    with open(episode_fpath,'r') as episode_file:
        transcript = episode_file.read()
    
    ## preprocess
    transcript = re.sub("[@$>%♪?,().0-9-\n'#*;:\"]",'',transcript)
    return transcript

def preprocess_loop():
    """ need better way to loop through directories"""

    for ch in ['FOX','MSNBC']:
        ch_dir = opj(raw_dir,ch)
        for show in os.listdir(ch_dir)[1:]:
            show_dir = opj(ch_dir,show)
            for transcript_fname in os.listdir(show_dir):
                
                # preprocess
                episode_fpath = opj(show_dir,transcript_fname)
                transcript = preprocess(episode_fpath)
                
                # write processed transcript to file
                episode_fname = episode_fpath.split('/')[-1]
                write_fpath = opj(output_dir,ch,episode_fname)
                with open(write_fpath,'w') as file:
                    file.write(transcript)
    return None











