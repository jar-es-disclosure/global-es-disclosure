#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@author: Yan LIN (jackylin2012@gmail.com)
"""

import pandas as pd
from scipy import spatial
import os
from glob import glob
import gensim
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
import logging
from scipy.spatial import distance

ESG_SENT_THRESHOLD = 100 # only filter ESG words > ESG_SENT_THRESHOLD

DATA_FOLDER = '<DATA_FOLDER>' # data folder in step1-preprocessing.py
RESULTS_FOLDER = '<RESULTS_FOLDER>'
PROCESSED_DF_PATH = '<PROCESSED_DF_PATH>' # PROCESSED_DF_PATH in step2-dump-sentences.py 
REPORT_LIST_FILE = 'short_listed_pdf.csv'

W2V_MODEL_PATH = '<W2V_MODEL_PATH>'  # RESULT_PATH in step4-run-word2vec.py
KEYWORD_FOLDER = '<KEYWORD_FOLDER>' # KEYWORD_FOLDER in gen-dictionary.py

PILLARS = ['Climate_Change','Natural_Resources','Pollution_Waste','Ecosystem','Human_Capital','Products_and_Customers','OtherStakeholders','general']

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

model = gensim.models.Word2Vec.load(os.path.join(W2V_MODEL_PATH,'all.word2vec'))

# read pillar dict
pillar_keywords = {}
for pillar in PILLARS:
    with open(os.path.join(KEYWORD_FOLDER,pillar + '.txt')) as f:
        pillar_keywords[pillar] = set([x.strip() for x in f.readlines() if len(x.strip()) > 0])

esg_words = set()
for _,v in pillar_keywords.items():
    esg_words.update(v)

def get_mean_vector(sent):
    global model
    words = [word for word in sent if word in model.wv]
    if len(words) >= ESG_SENT_THRESHOLD:
        return np.mean(model.wv[words],axis = 0)
    else:
        return None
    
def filter_and_encode_esg_sent(text):
    global esg_words
    esg_sents = []
    for sent in text:
        all_words = set(sent)
        if len(all_words.intersection(esg_words))>0:
            esg_sents.extend(sent)
    return get_mean_vector(esg_sents)

def run_get_stickyness():

    # Encode the sentences and save to processed_df_esg_sent_encoded folder for calculation
    process_files = glob(os.path.join(PROCESSED_DF_PATH,'*.feather'))
    for f_name in tqdm(process_files):
        df = pd.read_feather(f_name)
        df['dcn'] = df.f_name_list.str.split('/').str[-1].str.split('.').str[0]
        l_dcn = df.dcn.tolist()
    
        processed_df_path = os.path.join(DATA_FOLDER,'processed_df_esg_sent_encoded',f_name.split('/')[-1])
        if os.path.exists(processed_df_path):
            logging.info(f'{processed_df_path} exists. Skip.')
            continue
        
        logging.info(f'Processing {f_name}')
        start = time.time()
        
        if __name__ == '__main__':
            with multiprocessing.Pool(processes = 20) as pool:
                processed_sent = pool.map(filter_and_encode_esg_sent,df.processed_docs.tolist())
        processed_df = pd.DataFrame({'dcn':l_dcn,'esg_sent_vec':processed_sent})
        processed_df.to_feather(processed_df_path)
        logging.info(f'{processed_df_path} saved.')
        logging.info(f'Time elapsed: {time.time() - start}')

    # load all files
    process_files = glob(os.path.join(DATA_FOLDER,'processed_df_esg_sent_encoded','*.feather'))
    df_list = []
    for f_name in tqdm(process_files):
        df_list.append(pd.read_feather(f_name))

    df_stickyness = pd.concat(df_list,axis = 0)
    
    report_list_final = pd.read_csv(REPORT_LIST_FILE)
    report_list_final = report_list_final[report_list_final.ITEM2999.notna()] 
    
    df_stickyness = df_stickyness.merge(report_list_final[['fyear','dcn','OAPermID','periodEndDate','txt_file_size','OrgPermID','WsId','countryCode']],on = 'dcn')
    df_stickyness.sort_values(by = ['OAPermID','fyear','txt_file_size'] ,ascending=False,inplace = True)
    
    df_stickyness.txt_file_size.quantile(0.10) # drop the lower 10%
    # 3001.600000000006

    df_stickyness = df_stickyness.query('txt_file_size > 3000')
    df_stickyness.drop_duplicates(['OAPermID','fyear'],keep = 'first',inplace = True)
    
    df_stickyness['fyear_index'] = pd.to_datetime(df_stickyness['fyear'],format='%Y')
    df_stickyness.set_index('fyear_index',inplace = True)

    l1 = df_stickyness.groupby('OAPermID')['esg_sent_vec'].shift(1,freq=pd.DateOffset(years = 1))
    l1 = l1.reset_index()
    l1.rename(columns = {'esg_sent_vec':'esg_sent_vec_l1'},inplace = True)
    
    df_stickyness = df_stickyness.merge(l1,on = ['OAPermID', 'fyear_index'])
    df_stickyness.dropna(inplace = True) # may cause some data loss
    
    def get_stickyness(x,y):
        return 1 - spatial.distance.cosine(x,y)
    
    df_stickyness['stickyness'] = df_stickyness.apply(lambda x: get_stickyness(x.esg_sent_vec,x.esg_sent_vec_l1),axis = 1)

    cols = ['dcn', 'fyear_index', 'fyear', 'OAPermID','periodEndDate', 'stickyness','countryCode']
    df_stickyness[cols].to_stata(os.path.join(RESULTS_FOLDER,'stickyness.dta'))
    df_stickyness[cols].to_pickle(os.path.join(RESULTS_FOLDER,'stickyness.pickle'))


if __name__ == '__main__':
    run_get_stickyness()



