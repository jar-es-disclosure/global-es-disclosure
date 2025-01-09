#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Generating E&S Length and Specificity variables. 

@author: Yan LIN (jackylin2012@gmail.com)
"""

import os
import pandas as pd
from glob import glob
from tqdm import tqdm
import multiprocessing
import spacy
import logging
import time
import fire

RESULTS_FOLDER = '<RESULTS_FOLDER>'
KEYWORD_FOLDER = '<KEYWORD_FOLDER>' # KEYWORD_FOLDER in gen-dictionary.py

PILLARS = ['Climate_Change','Natural_Resources','Pollution_Waste','Ecosystem','Human_Capital','Products_and_Customers','OtherStakeholders','general']
PROCESSED_DF_FOLDER = '<PROCESSED_DF_PATH>' # PROCESSED_DF_PATH in step2-dump-sentences.py 

NER_TAGS = ['CARDINAL',
 'DATE',
 'EVENT',
 'FAC',
 'GPE',
 'LANGUAGE',
 'LAW',
 'LOC',
 'MONEY',
 'NORP',
 'ORDINAL',
 'ORG',
 'PERCENT',
 'PERSON',
 'PRODUCT',
 'QUANTITY',
 'TIME',
 'WORK_OF_ART']


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(RESULTS_FOLDER,'intensity_specificity.log'),mode='w')
    ]
)

for ner in NER_TAGS:
        logging.info(f'{ner}: {spacy.explain(ner)}')

NER_TAGS = set([f'[NER:{x}]' for x in NER_TAGS]) # Same as we recoded NER tags


def get_pillar_length(doc):
    """The total number of words from E&S-related sentences. E&S-related sentences are defined as those containing words from the E&S dictionary developed using Word2Vec.
    Args:
        doc (str): List[List[str]]

    Returns:
        dict: a dict representing a row
    """
    global pillar_keywords, PILLARS

    pillar_count = {}
    for pillar in PILLARS:
        pillar_count[pillar] = 0
        pillar_count[pillar+'_details'] = [] # hold results for verification
        
    pillar_count['doc_word_count'] = 0
    for sent in doc:     
        all_words = set(sent)
        for pillar in PILLARS:
            if len(all_words.intersection(pillar_keywords[pillar])) > 0: 
                pillar_count[pillar] += len(sent)
                
                details =  f"{' '.join(sent)} : {[w for w in sent if w in pillar_keywords[pillar]]}" # a new version
                pillar_count[pillar+'_details'].append(details)
                
        pillar_count['doc_word_count'] += len(sent)
    
    for pillar in PILLARS:
        pillar_count[pillar+'_details'] = ' ;@; '.join(pillar_count[pillar+'_details'])
    
    return pillar_count
       

def get_pillar_specificity(doc):
    """ The number of words in E&S-related sentences conveying specific names of persons, locations, and organizations; quantitative values in money values and percentages; and dates and times, scaled by the total number of words in E&S-related sentences.
    E&S-related sentences are defined as those containing words from the E&S dictionary developed using Word2Vec. Specific names are identified by Named entity recognition (NER).
    Args:
        doc (str): List[List[str]]

    Returns:
        dict: a dict representing a row
    """
    global esg_words

    results = {}
    results['esg_specificity'] = 0
    results['esg_specificity_details'] = []
    results['esg_word_count'] = 0
    results['esg_specificity_count'] = 0
    
    for sent in doc:
        all_words = set(sent)
        if len(all_words.intersection(esg_words))>0:
            results['esg_word_count'] += len(sent)
            details =  f'{sent} : {[w for w in sent if w in esg_words]}'
            results['esg_specificity_details'].append(details)
            for w in sent:
                if w in NER_TAGS:
                    results['esg_specificity_count'] += 1
    
    if results['esg_word_count'] > 0:
        results['esg_specificity'] = results['esg_specificity_count'] / results['esg_word_count'] 
    results['esg_specificity_details'] = ';@;'.join(results['esg_specificity_details'])
    return results
                    
def run_get_intensity_specificity():
    global pillar_keywords, esg_words

    # read keywords
    pillar_keywords = {}
    for pillar in PILLARS:
        with open(os.path.join(KEYWORD_FOLDER,pillar + '.txt')) as f:
            pillar_keywords[pillar] = set([x.strip() for x in f.readlines() if len(x.strip()) > 0])

    esg_words = set()
    for _,v in pillar_keywords.items():
        esg_words.update(v)
    
    processed_intensity_all = []
    processed_specificity_all = []
    process_files = glob(os.path.join(PROCESSED_DF_FOLDER,'*.feather'))
    for f_name in tqdm(process_files):
        df = pd.read_feather(f_name)
        df['dcn'] = df.f_name_list.str.split('/').str[-1].str.split('.').str[0]
        l_dcn = df.dcn.tolist()
        logging.info(f'Processing {f_name}')

        start = time.time()
        if __name__ == '__main__':
            with multiprocessing.Pool(processes = 20) as pool:
                processed_intensity = pool.map(get_pillar_length,df.processed_docs.tolist())
        processed_intensity = pd.DataFrame(processed_intensity)
        processed_intensity['dcn'] = l_dcn
        processed_intensity_all.append(processed_intensity)
        logging.info(f'Intensity calculated. Time elapsed: {time.time() - start}')
        
        logging.info('Intensity calculated.')
        
        if __name__ == '__main__':
            with multiprocessing.Pool(processes = 20) as pool:
                processed_specificity = pool.map(get_pillar_specificity,df.processed_docs.tolist())
        processed_specificity = pd.DataFrame(processed_specificity)
        processed_specificity['dcn'] = l_dcn
        processed_specificity_all.append(processed_specificity)
        logging.info(f'specificity calculated. Time elapsed: {time.time() - start}')
        
        
    processed_intensity_all = pd.concat(processed_intensity_all, axis=0)
    processed_specificity_all = pd.concat(processed_specificity_all, axis=0)

    processed_intensity_all.to_pickle(os.path.join(RESULTS_FOLDER,'intensity_DEBUG.pickle'))
    processed_specificity_all.to_pickle(os.path.join(RESULTS_FOLDER,'specificity_DEBUG.pickle'))

    processed_intensity_all.drop([f'{pillar}_details' for pillar in PILLARS],axis = 1).to_pickle(os.path.join(RESULTS_FOLDER,'intensity.pickle'))
    processed_specificity_all.drop('esg_specificity_details',axis = 1).to_pickle(os.path.join(RESULTS_FOLDER,'specificity.pickle'))

if __name__ == '__main__':
    fire.Fire(run_get_intensity_specificity)