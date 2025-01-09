#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@author: Yan LIN (jackylin2012@gmail.com)
"""

import logging
import os
from gensim.models import Phrases
from gensim.models.phrases import Phraser,ENGLISH_CONNECTOR_WORDS
from gensim.models.word2vec import LineSentence
import pickle

LOG_PATH = 'log'
DUMP_PATH = 'all_sentences'
NGRAM_PATH = 'bigram_trigram_model'

MIN_COUNT = 5
THRESHOLD = 1

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_PATH,'get_bigram_trigram.log')),
        logging.StreamHandler()
    ]
)

def run_bigram_trigram():
    out_file = os.path.join(DUMP_PATH,'all.txt')
    processed_sentences = LineSentence(out_file)
    ngram_file_path = os.path.join(NGRAM_PATH,f'all_bigram_trigram_{THRESHOLD}.pkl')
    if os.path.exists(ngram_file_path):
        logging.info("Bigram trigram model exists. Pass.. ")     
    else:
        logging.info('Building bigram and trigam...')
        
        bigram = Phrases(processed_sentences, min_count=MIN_COUNT, connector_words = ENGLISH_CONNECTOR_WORDS ,delimiter = "_", threshold=THRESHOLD)
        trigram = Phrases(bigram[processed_sentences], min_count=MIN_COUNT, connector_words = ENGLISH_CONNECTOR_WORDS ,delimiter = "_",threshold=THRESHOLD)
        logging.info('Done.')
        
        with open(ngram_file_path,'wb') as f:
            pickle.dump((bigram,trigram),f)
        logging.info(f"bigram trigram saved at: {ngram_file_path}")

    
if __name__ == '__main__':
    run_bigram_trigram()