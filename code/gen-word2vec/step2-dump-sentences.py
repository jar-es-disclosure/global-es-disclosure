#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@author: Yan LIN (jackylin2012@gmail.com)
"""


import pandas as pd
from glob import glob
import logging
import os

PROCESSED_DF_PATH = 'data/processed_df' # processed_df in the previous step
DUMP_PATH = 'all_sentences'
LOG_PATH = 'log'


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_PATH,'dumping.log')),
        logging.StreamHandler()
    ]
)


def run_dump_all_sentences():
    """
    Write all of the sentences to the hard disk for analysis. Cannot load them into memory. We will use LineSentence.
    """
    process_files = glob(os.path.join(PROCESSED_DF_PATH,'*.feather'))
    out_file = os.path.join(DUMP_PATH,'all.txt')
    logging.info(f"Dumping sentences to: {out_file}. {len(process_files)} .feather files found.")
    total_docs = 0
    total_sentences = 0
    with open(out_file,'w') as f:
        for f_name in process_files:
            df = pd.read_feather(f_name)
            for doc in df.processed_docs.tolist():
                for line in doc:
                    if len(line) == 0:
                        continue
                    f.write(' '.join(line) + '\n')
                total_sentences += len(doc)
            total_docs += df.shape[0]
            logging.info(f"Processing {f_name}. Total: documents: {total_docs}. Total sentences: {total_sentences}")  

if __name__ == '__main__':
    run_dump_all_sentences()
    
    
    
