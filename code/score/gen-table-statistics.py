#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Counting ESG related tables.

@author: Yan LIN (jackylin2012@gmail.com)
"""

import pandas as pd
import os
from docx import Document
from tqdm import tqdm
import logging
import re
from tqdm import tqdm
from glob import glob
from docx import Document
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from docx.table import _Cell

RESULTS_FOLDER = '<RESULTS_FOLDER>' 
KEYWORD_FOLDER = '<KEYWORD_FOLDER>' # KEYWORD_FOLDER in gen-dictionary.py
DATA_A_FOLDER = '<DATA_A_FOLDER>' # data too big to fit in one harddrive, so we have DATA_A_FOLDER and DATA_B_FOLDER
DATA_B_FOLDER = '<DATA_B_FOLDER>'

DEBUG = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)


PILLARS = ['Climate_Change','Natural_Resources','Pollution_Waste','Ecosystem','Human_Capital','Products_and_Customers','OtherStakeholders','general']

# read keywords
pillar_keywords = {}
for pillar in PILLARS:
    with open(os.path.join(KEYWORD_FOLDER,pillar + '.txt')) as f:
        pillar_keywords[pillar] = set([x.strip() for x in f.readlines() if len(x.strip()) > 0])


esg_words = set()

for _,v in pillar_keywords.items():
    esg_words.update(v)

def is_esg_sent(sent):
    for w in esg_words:
        if len(w.split('_')) == 1:
            if w in set(sent.split()):
                return True,w
        else:
            if (' '.join(w.split('_'))) in sent:
                return True,w
    return False,None


def pretify_text(t):
    rx = re.compile(r'\s+')
    return rx.sub(' ',t).strip()

def count_table(task):
    try:
        dcn = task['dcn']
        data = task ['data']
        
        doc = Document(data)
        
        l_table = doc.tables
        l_table_text = []

        for table in l_table:
            text = []
            
            for row in table.rows:
                cells = [_Cell(tc,table) for tc in row._tr.tc_lst]
                text.append(' '.join([cell.text for cell in cells]))
            text = ' '.join(text)
            l_table_text.append(text)
        
        df_table = pd.DataFrame({'text':l_table_text,'num_tbl':1}) # intermedia results each row is a table
        df_table['contain_esg'] = df_table.text.map(lambda x:is_esg_sent(x)[0])
        df_table['esg_flag'] = df_table.text.map(lambda x:is_esg_sent(x)[1])
        df_table = df_table.query('contain_esg == True')
        if DEBUG:
            return {'dcn':dcn,'table_count':df_table.num_tbl.sum(),'esg_text_tbl':'--@--'.join(df_table.text.tolist()),'esg_flag_tbl':'--@--'.join(df_table.esg_flag.tolist())}
        else:
            return {'dcn':dcn,'table_count':df_table.num_tbl.sum(),'esg_text_tbl':'','esg_flag_tbl':'--@--'.join(df_table.esg_flag.tolist())}
    
    except Exception as e:
        
        return {'dcn':dcn,'table_count':None,'esg_text_tbl':None,'esg_flag_tbl':None}

if __name__ == '__main__':
    os.makedirs(os.path.join(RESULTS_FOLDER,'visuals_tables','results'),exist_ok=True)

    logging.info(f'loading task...')
    if os.path.exists(os.path.join(RESULTS_FOLDER,'visuals_tables','file_list.dataframe')):
        logging.info(f'Cache file list...')
        df = pd.read_pickle(os.path.join(RESULTS_FOLDER,'visuals_tables','file_list.dataframe'))
    else:
        logging.info(f'compiling cache...')
        all_file_path = []
        
        for root, dirs, files in os.walk(DATA_A_FOLDER):
            for file in files:
                if file.endswith('.docx'):
                    all_file_path.append(os.path.join(root,file))
        df_dataA = pd.DataFrame({'docx_path':all_file_path})
        df_dataA['dcn'] = df_dataA.docx_path.map(lambda x:x.split('/')[-1].split('.')[0])

        all_file_path = []
        for root, dirs, files in os.walk(DATA_B_FOLDER):
            for file in files:
                if file.endswith('.docx'):
                    all_file_path.append(os.path.join(root,file))

        df_dataB = pd.DataFrame({'docx_path':all_file_path})
        df_dataB['dcn'] = df_dataB.docx_path.map(lambda x:x.split('/')[-1].split('.')[0])
        df = pd.concat([df_dataA,df_dataB],ignore_index=True)
        df.sort_values('docx_path',inplace=True)
        df.drop_duplicates('dcn',keep = 'first',inplace=True)
        df.to_pickle(os.path.join(RESULTS_FOLDER,'visuals_tables','file_list.dataframe'))
    
    existing_df = []
    for f_name in glob(os.path.join(RESULTS_FOLDER,'visuals_tables','results','visuals_*.pickle')):
        existing_df.append(pd.read_pickle(f_name))
    
    if len(existing_df) > 0:
        existing_df = pd.concat(existing_df,axis = 0)
        
        df['dcn'] = df.docx_path.str.split('/').str[-1].str.split('.').str[0]
        print(df.shape)
        df = df[~df.dcn.isin(existing_df.dcn)]
        print(df.shape)
    
    # Processing... 
    files_for_processing = df.docx_path.tolist() 
    CHUNKSIZE = 5000  
    TIMEOUT = 120

    i = 0
    processed_df_path = os.path.join(RESULTS_FOLDER,'visuals_tables','results',f'visuals_{i}.pickle')
    while os.path.exists(processed_df_path):
        i += 1
        processed_df_path = os.path.join(RESULTS_FOLDER,'visuals_tables','results',f'visuals_{i}.pickle')
    
    
    logging.info(f'Total files to process: {len(files_for_processing)}')
    for f_name_list_chunk in [files_for_processing[j:j+CHUNKSIZE] for j in range(0,len(files_for_processing),CHUNKSIZE)]:
        
        processed_df_path = os.path.join(RESULTS_FOLDER,'visuals_tables','results',f'visuals_{i}.pickle')
        
        logging.info(f'Calculating file {i}.feather')
        logging.info(f'Reading files')
        tasks = []
        
        
        for f_name in tqdm(f_name_list_chunk):
            try:
                dcn = f_name.split('/')[-1].split('.')[0]
                
                tasks.append({'dcn':dcn,'data':f_name})
            except Exception as e:
                print(e)
        

        
        if __name__ == '__main__':
            
            print(f'{len(tasks)}')
            with ProcessPool(max_workers = 15) as pool:
                future = pool.map(count_table, tasks, timeout=TIMEOUT)
                
                
                print('starting')
                iterator = future.result()
                print('ending...')
                
            results = []
            while True:
                try:
                    result = next(iterator)
                    results.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    
                    results.append(None)
                except ProcessExpired as error:
                    print("%s. Exit code: %d" % (error, error.exitcode))
                except Exception as error:
                    print("function raised %s" % error)
                    print(error.traceback)  # Python's traceback of remote process
        
        logging.info(f'End of multiprocessing.')
        print(f'{len(results)}')
        x = [r for r in results if r is not None] # filter out 
        
        logging.info(f'Done. Writing results. Number Results: {len(x)}')
        missing_set = set([task['dcn'] for task in tasks]) - set([r['dcn'] for r in x if r is not None])
        logging.info(f'Missing dcn: {missing_set}')
        del tasks

        df_results = pd.DataFrame(x)
        df_results.to_pickle(os.path.join(RESULTS_FOLDER,'visuals_tables','results',f'visuals_{i}.pickle'))
        
        logging.info('Done.')
        del df_results
        i += 1
        
    result_dfs = []
    logging.info('combining results')
    for f_name in glob(os.path.join(RESULTS_FOLDER,'visuals_tables','results','visuals_*.pickle')):
        result_dfs.append(pd.read_pickle(f_name))
    final_df = pd.concat(result_dfs,axis = 0)
    final_df.to_pickle(os.path.join(RESULTS_FOLDER,'visuals_tables','results','visual.pickle'))
    final_df[['dcn', 'pic_count', 'table_count']].to_stata(os.path.join(RESULTS_FOLDER,'visuals','results','visual.dta'),write_index= False)
    logging.info(f'Rows: {final_df.shape[0]}')
    logging.info(f'{final_df.head()}')