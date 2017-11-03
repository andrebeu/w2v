from os.path import join as opj
import os, json, re
import random

import enchant

""" misc functions: count number of incorrect spelling
"""

print('-------------START-------------')


# channel : MSNBC, FOX
# show: foxandfriends,hannity,...
# episode: hannity031, hannity032...

channel = 'FOX'
output_dir = opj('data','output')
channel_dir = opj('data',channel)
preprocessed_dir = opj(output_dir,'preprocessed',channel)


def count_incorrect_spelling():
    """change:
        input: path to folder containing transcripts """

    dictionary = enchant.Dict('en_US')
    D = {}
    for show_dir in os.listdir(channel_dir)[1:]:
        show_name = show_dir.split('/')[-1]

        for transcript_fname in os.listdir(opj(channel_dir,show_dir)):
            episode_fpath = opj(channel_dir,show_dir,transcript_fname)
            episode_id = episode_fpath.split('/')[-1]
            
            print(episode_id)
            with open(episode_fpath) as file:
                transcript = file.read()

            n_words = len(transcript.split())

            misspell_count = 0
            # count incorrect spelling
            for word in transcript.split():
                if dictionary.check(word) == False:
                    misspell_count += 1

            try: # count proportion of incorrectly spelled words
                misspell_proportion = misspell_count/n_words
            except: # division by zero
                assert n_words == 0, print('SEE',show_id)
                misspell_proportion = "ZERO wordcount"

            D[episode_id] = misspell_proportion

        # write incorrect spell count
        fpath = opj(output_dir,'spell_counts','spelling_%s.txt'%show_name)
        with open(fpath,'w') as file:
            file.write(json.dumps(D))


count_incorrect_spelling()