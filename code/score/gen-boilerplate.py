#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Calculation is based on Lang and Stice-Lawrence 2015 JAE paper, Session 3.2 and Section A.2. Boilerplate. provide the details.

(1) For each country, identify tetragrams (four-word phrases) that appear in 30% of the documents by country
    and identify tetragrams that appear on average at least 5 times per document
    
(2) Check the previous tetragrams identified in (1), only retaining those appearing in 60% of the documents by country
(3) By Country, removing tegragrams appearing in 80% of the documents, and 
    removing tegragrams appearing in 75% of all the documents

@author: Yan LIN (jackylin2012@gmail.com)
"""

import pickle
from collections import Counter
from nltk import ngrams
from nltk import sent_tokenize
import os
from tqdm import tqdm
import logging
import multiprocessing
import pandas as pd
from glob import glob
import itertools
import re

# start from raw data
DATA_FOLDER = "<DATA_FOLDER>" # data folder in step1-preprocessing.py
REPORT_LIST_FILE = "short_listed_pdf.csv"
KEYWORD_FOLDER = '<KEYWORD_FOLDER>' # KEYWORD_FOLDER in gen-dictionary.py

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

CHUNKSIZE = 5000

PILLARS = ['Climate_Change','Natural_Resources','Pollution_Waste','Ecosystem','Human_Capital','Products_and_Customers','OtherStakeholders','general']

pillar_keywords = {}
for pillar in PILLARS:
    with open(os.path.join(KEYWORD_FOLDER,pillar + '.txt')) as f:
        pillar_keywords[pillar] = set([x.strip() for x in f.readlines() if len(x.strip()) > 0])

esg_words = set()
for _,v in pillar_keywords.items():
    esg_words.update(v)
    


# ---------------------------------------------------------------
# Step 0. We cannot reuse the previously tokenized sentences because we preprocess them by concatenating phrases with "_".
# Here we only need to sentensize the documents
# Data are saved in <DATA_FOLDER>/esg_df/*.feather
# ---------------------------------------------------------------

# Load file list
with open(os.path.join(DATA_FOLDER,'cache','files_for_processing.pkl'),'rb') as f:
    files_for_processing = pickle.load(f)

def sentensize(text):
    return [sent.lower() for sent in sent_tokenize(text)]

i = 0
for f_name_list_chunk in [files_for_processing[i:i+CHUNKSIZE] for i in range(0,len(files_for_processing),CHUNKSIZE)]:
    
    processed_df_path = os.path.join(DATA_FOLDER,'esg_df',f'{i}.feather')
    
    if os.path.exists(processed_df_path):
        logging.info(f'{processed_df_path} exists. Skip')
    else:
        docs = []
        logging.info('Loading Files')
        for f_name in tqdm(f_name_list_chunk):
            with open(f_name) as f:
                docs.append(f.read())
        logging.info('Files loaded to memory...')
        logging.info('sententize document...')
        if __name__ == '__main__':
            with multiprocessing.Pool(processes = 20) as pool:
                processed_docs = pool.map(sentensize,docs) # use pool.imap may lead to performance loss.
        logging.info('Making dataframe...')
        processed_docs_df = pd.DataFrame({"f_name_list":f_name_list_chunk,"processed_docs":processed_docs})
        logging.info('Saving data frame...')
        
        processed_docs_df.to_feather(processed_df_path)
        logging.info(f'Saved to {processed_df_path}.')
        
        del processed_docs_df
        del processed_docs
        del docs
        
    i += 1
    

# ---------------------------------------------------------------
# (1) For each country, identify tetragrams (four-word phrases) that appear in 30% of the documents by country
#    and identify tetragrams that appear on average at least 5 times per document

# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Step 1.1 Count tetragrams 
# Data are saved in <DATA_FOLDER>/<countryCode>.pkl
# Each pkl serialize the counter object (a dict containing the counts of tetragrams) and the doc_count storing how many documents in the country
# ---------------------------------------------------------------

logging.info('Calculating tetragrams...')
report_list_final = pd.read_csv(REPORT_LIST_FILE)
report_list_final = report_list_final[report_list_final.ITEM2999.notna()] 

# gent country count
os.makedirs(os.path.join(DATA_FOLDER,'esg_boilerplate'),exist_ok=True)
for code in ['IN', 'AU', 'GB', 'SG', 'SE', 'LK', 'DK', 'CA', 'FR', 'MY', 'HK',
       'JP', 'BE', 'FI', 'NZ', 'CH', 'DE', 'NL', 'IT', 'PK', 'GR', 'TR',
       'AT', 'NG', 'CN', 'IE', 'ZA', 'RU', 'NO', 'ID']:
    if os.path.exists(os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}.pkl')):
        continue
    logging.info(f'Processing {code}..')
    counter = Counter()
    doc_count = 0
    for f_name in tqdm(glob(os.path.join(DATA_FOLDER,'esg_df','*.feather'))):
        print(f'processing {f_name}')
        processed_docs_df = pd.read_feather(f_name)
        processed_docs_df['dcn'] = processed_docs_df.f_name_list.str.split('/').str[-1].str.split('.').str[0]
        processed_docs_df = processed_docs_df.merge(report_list_final[['dcn','countryCode']],on = 'dcn')
        processed_docs_df = processed_docs_df.query(f"countryCode == '{code}'")
        for _, v in processed_docs_df.iterrows():
            dcn = v['dcn']
            sents = v['processed_docs']
            countryCode = v['countryCode']
            ngrams_set = set() # used for removing repeating tetragrams in a document
           
            if len(sents) == 0:
               continue
            for sent in sents:
                ngrams_set.update(ngrams(sent.split(),4)) # only consider the ngrams once in one document, because we want to see the percentage of documents containing the tetragrams
            counter.update(ngrams_set)
            doc_count = doc_count + 1
    logging.info(f'dumpping file')
    if code == 'IN' or code == 'GB':
        counter = Counter({k:c for k,c in counter.items() if c>=1000}) # added later to prune results, when country IN and GB cause the program to crash. this would not affect the results.
    
    with open(os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}.pkl'),'wb') as f: 
        pickle.dump((counter,doc_count),f)
    del counter
    del doc_count
    

# ---------------------------------------------------------------
# Step 1.2 Calculate frequencies of all n grams in all documents.
# Note that different from Step 1.1, we do not use set because we want to see actualy how many times they appear

# Data are saved in <DATA_FOLDER>/esg_boilerplate/{code}_counter_for_all.pkl
# Each pkl serialize the counter object (a dict containing the counts of tetragrams)
# ---------------------------------------------------------------


for code in ['IN', 'AU', 'GB', 'SG', 'SE', 'LK', 'DK', 'CA', 'FR', 'MY', 'HK',
       'JP', 'BE', 'FI', 'NZ', 'CH', 'DE', 'NL', 'IT', 'PK', 'GR', 'TR',
       'AT', 'NG', 'CN', 'IE', 'ZA', 'RU', 'NO', 'ID']:
    
    if os.path.exists(os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}_counter_for_all.pkl')):
        logging.info(f'{code}_counter_for_all.pkl exists. Skip.')
        continue
    logging.info(f'Processing {code}..')
    counter = Counter()
    
    for f_name in tqdm(glob(os.path.join(DATA_FOLDER,'esg_df','*.feather'))):
        print(f'processing {f_name}')
        processed_docs_df = pd.read_feather(f_name)
        processed_docs_df['dcn'] = processed_docs_df.f_name_list.str.split('/').str[-1].str.split('.').str[0]
        processed_docs_df = processed_docs_df.merge(report_list_final[['dcn','countryCode']],on = 'dcn')
        processed_docs_df = processed_docs_df.query(f"countryCode == '{code}'")
        for _, v in processed_docs_df.iterrows():
            dcn = v['dcn']
            sents = v['processed_docs']
            countryCode = v['countryCode']
           
            if len(sents) == 0:
               continue
            for sent in sents:
                counter.update(ngrams(sent.split(),4))
            
    if code == 'IN' or code == 'GB':
        counter = Counter({k:c for k,c in counter.items() if c>=1000}) # added later when country IN and GB cause the program to crash. this would not affect the results.
    with open(os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}_counter_for_all.pkl'),'wb') as f: 
        pickle.dump((counter),f)
    del counter

# get number of documents 
doc_count = 0
for f_name in tqdm(glob(os.path.join(DATA_FOLDER,'esg_df','*.feather'))):
    print(f'processing {f_name}')
    processed_docs_df = pd.read_feather(f_name)
    processed_docs_df['len_sent'] = processed_docs_df.processed_docs.map(len)
    processed_docs_df = processed_docs_df[processed_docs_df.len_sent > 0]
    doc_count += processed_docs_df.shape[0]
    
    
with open(os.path.join(DATA_FOLDER,'esg_boilerplate',f'all_doc_count.pkl'),'wb') as f: 
    pickle.dump((doc_count),f)


# ---------------------------------------------------------------
# Step 1.3 add the counters  by country, this counter will be used to test if the tetragram appears on average 5 times per document
# ---------------------------------------------------------------

total_counter = Counter()
for code in ['IN', 'AU', 'GB', 'SG', 'SE', 'LK', 'DK', 'CA', 'FR', 'MY', 'HK',
       'JP', 'BE', 'FI', 'NZ', 'CH', 'DE', 'NL', 'IT', 'PK', 'GR', 'TR',
       'AT', 'NG', 'CN', 'IE', 'ZA', 'RU', 'NO', 'ID']:
    logging.info(f'reading {code}')
    with open(os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}_counter_for_all.pkl'),'rb') as f: 
        counter = pickle.load(f)
    
    counter = Counter({k:c for k,c in counter.items() if c>=1000}) # pruning, asssuming that tegragrams that are less than 1000 has no chance of being a candidate, drop them to reduce memory

    total_counter.update(counter)


with open(os.path.join(DATA_FOLDER,'esg_boilerplate',f'all_doc_count.pkl'),'rb') as f: 
    doc_count = pickle.load(f)
greater_tetragrams = [k for k,c in total_counter.items() if c >= 5*doc_count]


# save greater_tetragrams
with open(os.path.join(DATA_FOLDER,'esg_boilerplate',f'greater_tetragrams.pkl'),'wb') as f: 
    pickle.dump(greater_tetragrams,f)
    
print(len(greater_tetragrams))

    

# ---------------------------------------------------------------
# Step 1.5 add all tetragrams in all countries to identify tegragrams appears in 75% of all the documents. 
# These will be removed later by country.
# ---------------------------------------------------------------


total_counter_doc = Counter()
total_num_docs = 0
for code in ['IN', 'AU', 'GB', 'SG', 'SE', 'LK', 'DK', 'CA', 'FR', 'MY', 'HK',
       'JP', 'BE', 'FI', 'NZ', 'CH', 'DE', 'NL', 'IT', 'PK', 'GR', 'TR',
       'AT', 'NG', 'CN', 'IE', 'ZA', 'RU', 'NO', 'ID']:
    logging.info(f'reading {code}')
    with open(os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}.pkl'),'rb') as f: 
       counter,doc_count = pickle.load(f)
    counter = Counter({k:c for k,c in counter.items() if c>=1000})
    
    total_counter_doc.update(counter)
    total_num_docs = total_num_docs + doc_count
    del counter
    

with open(os.path.join(DATA_FOLDER,'esg_boilerplate',f'total_counter_for75.pkl'),'wb') as f:
    pickle.dump((total_counter_doc,total_num_docs),f)


# ----------------
# Step 2
# In each country, pick tetragrams appear in more than 30% of the documents, or appear an average of 5 times in each of the countries.
# -----------------


# load tegragrams appear on average 5 times in each of the documents. 
# there are 18 in total

with open(os.path.join(DATA_FOLDER,'esg_boilerplate',f'greater_tetragrams.pkl'),'rb') as f: 
    greater_tetragrams = pickle.load(f)

with open(os.path.join(DATA_FOLDER,'esg_boilerplate',f'total_counter_for75.pkl'),'rb') as f:
    total_counter_doc,total_num_docs = pickle.load(f)

# short list tegragrams appear in more than 75% of the documents
greater_75 = set([k for k,c in total_counter_doc.items() if c >= 0.75 * total_num_docs])

for code in ['IN', 'AU', 'GB', 'SG', 'SE', 'LK', 'DK', 'CA', 'FR', 'MY', 'HK',
       'JP', 'BE', 'FI', 'NZ', 'CH', 'DE', 'NL', 'IT', 'PK', 'GR', 'TR',
       'AT', 'NG', 'CN', 'IE', 'ZA', 'RU', 'NO', 'ID']:
    with open(os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}.pkl'),'rb') as f: 
       counter,doc_count = pickle.load(f)
    
    # (1) For each country, identify tetragrams (four-word phrases) that appear in 30% of the documents by country.
    # and identify tetragrams that occur on average at least 5 times per document.

    # shortlist tetragrams appear in 30% of the documents in the country
    shortlisted_tetragrams = set([k for k,c in counter.items() if c >= 0.3 * doc_count])
    print(f'{len(shortlisted_tetragrams)} greater than 30%')
    
    # plus those appear on average at least 5 times per document
    shortlisted_tetragrams.update(greater_tetragrams)
    print(f'added on average 5, now number of tetragrams: {len(shortlisted_tetragrams)}')
    
    #-----
    # (2) Check the previous tetragrams identified in (1), only retaining those appearing in 60% of the documents by country

    to_keep = set([k for k,c in counter.items() if c >= 0.6 * doc_count])
    print(f'To keep 60%: {len(to_keep)}')
    
    shortlisted_tetragrams = set([t for t in shortlisted_tetragrams if t in to_keep])
    print(f'after keeping 60% {len(shortlisted_tetragrams)}')
    
    # (3) By Country, removing tegragrams appearing in 80% of the documents, and 
    #     removing tegragrams appearing in 75% of all the documents
    to_drop = set([k for k,c in counter.items() if c >= 0.8 * doc_count])
    to_drop.update(greater_75)
    
    print(f'dropping grater than 80% or appear in 75% of the documents {len(to_drop)}')
    shortlisted_tetragrams = set([t for t in shortlisted_tetragrams if t not in to_drop])
    
    print(f"Dumpping to: {os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}_shortlisted.pkl')}")
    with open(os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}_shortlisted.pkl'),'wb') as f:
        pickle.dump(shortlisted_tetragrams,f)
    print('Done.')
    
    print(f"Dumpping to: {os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}_shortlisted.xlsx')}")
    df = pd.DataFrame({'tegragram':list(shortlisted_tetragrams)})
    df['tegragram'] = df['tegragram'].map(lambda x: ' '.join(x))
    df.to_excel(os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}_shortlisted.xlsx'),index = False)
    print('Done.')
    del counter
        

# --------------------------
# Step 3 filter sentences
# --------------------------

def is_esg_sent(sent):
    for w in esg_words:
        if len(w.split('_')) == 1:
            if w in set(sent.split()):
                return True,w
        else:
            if ' '.join(w.split('_')) in sent:
                
                return True,w
    return False,None

def is_boilerplate_sent(sent):
    if len(flags.intersection(ngrams(sent.split(),4))) > 0:
        return True, ','.join([' '.join(x) for x in flags.intersection(ngrams(sent.split(),4))])
    else:
        return False,None


rx = re.compile(r'\n+')
def pretify_sent(sent):
    return rx.sub(' ',sent).strip()

def filer_esg_boilderplate(sents):
    final = []
    for sent in sents:
        is_esg, flag_esg = is_esg_sent(sent)
        is_boilderplage, flag_boilderplate = is_boilerplate_sent(sent)
        if is_esg and is_boilderplage:
            final.append((pretify_sent(sent),flag_boilderplate,flag_esg))
    return final

def filer_esg_sents(sents):
    return [pretify_sent(sent) for sent in sents if is_esg_sent(sent)[0]]

for code in ['IN', 'AU', 'GB', 'SG', 'SE', 'LK', 'DK', 'CA', 'FR', 'MY', 'HK',
       'JP', 'BE', 'FI', 'NZ', 'CH', 'DE', 'NL', 'IT', 'PK', 'GR', 'TR',
       'AT', 'NG', 'CN', 'IE', 'ZA', 'RU', 'NO', 'ID']:
    
    logging.info(f'Processing {code}..')
    if os.path.exists(os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}_boilderplate_sent.pkl')):
        logging.info(f'{code} Skip.')
        continue
    with open(os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}_shortlisted.pkl'),'rb') as f:
        flags = pickle.load(f)
    
    final_dfs = []
    esg_boilderplate = []
    for f_name in tqdm(glob(os.path.join(DATA_FOLDER,'esg_df','*.feather'))):
        print(f'processing {f_name}')
        processed_docs_df = pd.read_feather(f_name)
        processed_docs_df['dcn'] = processed_docs_df.f_name_list.str.split('/').str[-1].str.split('.').str[0]
        processed_docs_df = processed_docs_df.merge(report_list_final[['dcn','countryCode']],on = 'dcn')
        processed_docs_df = processed_docs_df.query(f"countryCode == '{code}'")
        docs = processed_docs_df['processed_docs']
        
        if __name__ == '__main__':
            with multiprocessing.Pool(processes = 20) as pool:
                processed_docs = pool.map(filer_esg_boilderplate,docs) 
                esg_sents = pool.map(filer_esg_sents,docs) 
        esg_boilderplate.extend(set(itertools.chain(*processed_docs)))
        final_dfs.append(pd.DataFrame({'dcn':processed_docs_df.dcn,'processed_docs':processed_docs,'esg_sents':esg_sents}))
        
    esg_boilderplate = list(dict.fromkeys(esg_boilderplate)) # remove duplicates
    
    final_df = pd.concat(final_dfs,axis = 0)   
    final_df['count_esg_boilerplate'] = final_df['processed_docs'].map(len)
    final_df['count_esg_sents'] = final_df['esg_sents'].map(len) 
    with open(os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}_boilderplate_sent.pkl'),'wb') as f: 
        pickle.dump(final_df,f)
    pd.DataFrame(esg_boilderplate,columns = ['sent','tetragram','esg_keyword']).to_excel(os.path.join(DATA_FOLDER,'esg_boilerplate',f'{code}_boilderplate_sent_view.xlsx'))

