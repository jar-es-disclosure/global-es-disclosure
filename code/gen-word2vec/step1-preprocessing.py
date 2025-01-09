#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@author: Yan LIN (jackylin2012@gmail.com)
"""

import os
from tqdm import tqdm
import pandas as pd
import pickle
import logging
import multiprocessing
import spacy 
from datetime import datetime
import re

DATA_FOLDER = "data" # Path for storing the cache and the pre-processed data.
LOG_PATH = 'log' # log path
REPORT_LIST_FILE = 'short_listed_pdf.csv' # list of the reports that we need to analyze
TXT_FOLDER = "docx_to_txt" # TARGET_FOLDER in parse to txt step. 

CHUNKSIZE = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_PATH, 'preprocessing_'+ f"{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}" + '.log')),
        logging.StreamHandler()
    ]
)


class Sententizer():
    '''
     # Merge compounds identified by dependency parser into one token.
     # Multi-word expressions are left for gensim to generate bigram and trigram.
     # Lemmatize all words.
     # Replace entities with their types.
     # Remove punctuation, do not remove stop words and single letter words because they can be used in bigram and trigram recognition.
     # Change to lower case.
    '''
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = 30000000
        self.set_not_replace = set(['scope',
                                    'scope_1',
                                    'scope_2',
                                    'scope_3'])
    def sententize(self,text):
        if text is None:
            return []
        text = re.sub(r"\s+", " ", text) # replace multiple spaces and tabs
        
        doc = self.nlp(text)
        
        spans = [] # collect list of spans to be combined. There are two: entities and compounds. Both needs to be combined with _
        for tok in doc:
            if tok.dep_=="compound":
                spans.append(doc[tok.i:tok.head.i+1])
                
        for ent in doc.ents:
            spans.append(doc[ent.start:ent.end])
            
        spans = spacy.util.filter_spans(spans) # resolve spans overlaps to the longest. https://spacy.io/api/top-level
        with doc.retokenize() as retokenizer:
            for span in spans:
                span_text_list = []
                for w in span:
                    span_text_list.append(w.lemma_)
                retokenizer.merge(span,attrs = {"LEMMA":'_'.join(span_text_list)})
        
        return [self._lemmatize_and_replace_entity(sent) for sent in doc.sents if sent.text.strip()] 
        
        
    def _lemmatize_and_replace_entity(self,doc):
          
        result = []
        for t in doc:
            if (not t.is_punct) and (len(t.text.strip()) > 0):
                if (not t.ent_type_) or (t.lemma_.strip().lower() in self.set_not_replace):
                    temp = t.lemma_.strip().lower()
                    temp = temp.replace('_-_','-') # turn 'risk_-_return' back to 'risk-return' 
                    temp = re.sub(r"_+",'_', temp)# replace multiple __ to only one
                    result.append(temp)
                else:
                    result.append('[NER:' + t.ent_type_ + ']')
                    
        return result
    
def run_preprocessing():

    logging.info('Preprocessing starts.')
    logging.info('Collecting file names...')
    
    if os.path.exists(os.path.join(DATA_FOLDER,'cache','files_for_processing.pkl')):
        with open(os.path.join(DATA_FOLDER,'cache','files_for_processing.pkl'),'rb') as f:
            logging.info('files_for_processing cache found. Load it.')
            files_for_processing = pickle.load(f)
    else:
            
        all_file_path = []
        for root, dirs, files in os.walk(TXT_FOLDER):
            for file in files:
                if file.endswith('.txt'):
                    all_file_path.append(os.path.join(root,file))
                    
        all_txt_dcn = [x.split('/')[-1].replace('.txt','') for x in all_file_path]
                
        report_list_final = pd.read_csv(REPORT_LIST_FILE)
        report_list_final = report_list_final[report_list_final.ITEM2999.notna()]       
        
        valid_dcn = set(report_list_final.dcn)
        all_txt_dcn = set(all_txt_dcn)
        
        dcn_for_processing = valid_dcn.intersection(all_txt_dcn)
        
        files_for_processing = [x for x in all_file_path if x.split('/')[-1].replace('.txt','') in dcn_for_processing]
        with open(os.path.join(DATA_FOLDER,'cache','files_for_processing.pkl'),'wb') as f:
            pickle.dump(files_for_processing,f)
            
    logging.info(f'No. files for processing: {len(files_for_processing)}')
    
    sent_parser = Sententizer()
    i = 0
    
    for f_name_list_chunk in [files_for_processing[i:i+CHUNKSIZE] for i in range(0,len(files_for_processing),CHUNKSIZE)]:
        
        processed_df_path = os.path.join(DATA_FOLDER,'processed_df',f'{i}.feather')
        
        if os.path.exists(processed_df_path):
            logging.info(f'{processed_df_path} exists. Skip')
        else:
            docs = []
            logging.info('Loading Files')
            for f_name in tqdm(f_name_list_chunk):
                with open(f_name) as f:
                    docs.append(f.read())
            logging.info('Files loaded to memory.')
            logging.info('sententize and replace entities...')
            if __name__ == '__main__':
                with multiprocessing.Pool(processes = 20) as pool:
                    processed_docs = pool.map(sent_parser.sententize,docs) 
            logging.info('Making dataframe.')
            processed_docs_df = pd.DataFrame({"f_name_list":f_name_list_chunk,"processed_docs":processed_docs})
            logging.info('Saving data frame.')
            
            processed_docs_df.to_feather(processed_df_path)
            logging.info(f'Saved to {processed_df_path}')
            
            del processed_docs_df
            del processed_docs
            del docs
            
        i += 1
        
if __name__ == '__main__':
    run_preprocessing()
    
            
    

