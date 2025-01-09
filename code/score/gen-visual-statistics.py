#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Counting ESG related images. Tables are counted separately. 

@author: Yan LIN (jackylin2012@gmail.com)
"""
import pandas as pd
import multiprocessing
import os
from tqdm import tqdm
import logging
from bs4 import BeautifulSoup
import zipfile
import re
from glob import glob
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

RESULTS_FOLDER = '<RESULTS_FOLDER>'
KEYWORD_FOLDER = '<KEYWORD_FOLDER>' # KEYWORD_FOLDER in gen-dictionary.py
DATA_A_FOLDER = '<DATA_A_FOLDER>' # data too big to fit in one harddrive, so we have DATA_A_FOLDER and DATA_B_FOLDER
DATA_B_FOLDER = '<DATA_B_FOLDER>'
REPORT_LIST_FILE = 'short_listed_pdf.csv'

CHUNKSIZE = 5000
TIMEOUT = 3000

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

def find_previous_words(element, threshold = 300):
    '''
    find previouls <threshold> words

    ''' 
    
    results = ''
    if element.previous_siblings is not None:
        l_text = [x.text for x in element.previous_siblings]
        l_text.reverse()
        l_text = ' '.join(l_text)
        results = ' '.join(l_text.split()[-threshold:])
    
    while (element.parent is not None) and (results == ''):
        element = element.parent
        l_text = [x.text for x in element.previous_siblings]
        l_text.reverse()
        l_text = ' '.join(l_text)
        results = ' '.join(l_text.split()[-threshold:])
    return results


def find_next_words(element, threshold = 300):
    '''
    find previouls  <threshold>  words

    '''
    results = ''
    if element.next_siblings is not None:
        l_text = [x.text for x in element.next_siblings]
        l_text = ' '.join(l_text)
        results = ' '.join(l_text.split()[:threshold])
    
    while (element.parent is not None) and (results == ''):
        element = element.parent
        l_text = [x.text for x in element.next_siblings]
        l_text = ' '.join(l_text)
        results = ' '.join(l_text.split()[:threshold])
    return results


def count_visuals(task):
    try:
        
        dcn = task['dcn']
        data = task ['data']
        
        soup = BeautifulSoup(data,'lxml-xml')
                
        l_pic = soup.find_all('pic:pic')
        
        
        l_pic_text = [find_previous_words(x) + ' ' + find_next_words(x) for x in l_pic]
                
        df_pic = pd.DataFrame({'text':l_pic_text,'num_pic':1}) # intermedia results each row is a table
        
        
        df_pic['contain_esg'] = df_pic.text.map(lambda x:is_esg_sent(x)[0])
        df_pic['esg_flag'] = df_pic.text.map(lambda x:is_esg_sent(x)[1])
                
        df_pic = df_pic.query('contain_esg == True')
        
        
        if DEBUG:
            return {'dcn':dcn,'pic_count':df_pic.num_pic.sum(),'esg_text_pic':'--@--'.join(df_pic.text.tolist()),'esg_flag_pic':'--@--'.join(df_pic.esg_flag.tolist())}
        else:
            return {'dcn':dcn,'pic_count':df_pic.num_pic.sum(),'esg_text_pic':'','esg_flag_pic':'--@--'.join(df_pic.esg_flag.tolist())}
        
    except Exception as e:
        logging.info(e)
        return {'dcn':dcn,'pic_count':None,'esg_text_pic':None,'esg_flag_pic':None}

if __name__ == '__main__':
    os.makedirs(os.path.join(RESULTS_FOLDER,'visuals','results'),exist_ok=True)

    logging.info(f'loading task...')
    if os.path.exists(os.path.join(RESULTS_FOLDER,'visuals','file_list.dataframe')):
        logging.info(f'Cache file list...')
        df = pd.read_pickle(os.path.join(RESULTS_FOLDER,'visuals','file_list.dataframe'))
       
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

        report_list_final = pd.read_csv(REPORT_LIST_FILE)
        report_list_final.shape
        report_list_final = report_list_final[report_list_final.ITEM2999.notna()]
        df = df[df.dcn.isin(report_list_final.dcn)]
        df.to_pickle(os.path.join(RESULTS_FOLDER,'visuals','file_list.dataframe'))
    
    existing_df = []
    for f_name in glob(os.path.join(RESULTS_FOLDER,'visuals','results','visuals_*.pickle')):
        existing_df.append(pd.read_pickle(f_name))
    if len(existing_df) > 0:
        existing_df = pd.concat(existing_df,axis = 0)
    
    
    df['dcn'] = df.docx_path.str.split('/').str[-1].str.split('.').str[0]
    logging.info(df.shape)
    if type(existing_df) is not list:
        logging.info(f'dropping:{existing_df.shape[0]} processed files dcns. ')
        df = df[~df.dcn.isin(existing_df.dcn)]
    logging.info(df.shape)
    
    files_for_processing = df.docx_path.tolist() 
    
    i = 0
    processed_df_path = os.path.join(RESULTS_FOLDER,'visuals','results',f'visuals_{i}.pickle')
    while os.path.exists(processed_df_path):
        i += 1
        processed_df_path = os.path.join(RESULTS_FOLDER,'visuals','results',f'visuals_{i}.pickle')
    
    logging.info(f'Total files to process: {len(files_for_processing)}')
    for f_name_list_chunk in [files_for_processing[j:j+CHUNKSIZE] for j in range(0,len(files_for_processing),CHUNKSIZE)]:
        
        processed_df_path = os.path.join(RESULTS_FOLDER,'visuals','results',f'visuals_{i}.pickle')
        
        logging.info(f'Calculating file {i}.feather')
        logging.info(f'Reading files')
        tasks = []
           
        for f_name in tqdm(f_name_list_chunk):
            try:
                dcn = f_name.split('/')[-1].split('.')[0]
                with zipfile.ZipFile(f_name) as zfp:
                    with zfp.open('word/document.xml') as fp:
                        data = fp.read()
                tasks.append({'dcn':dcn,'data':data})
            except Exception as e:
                print(e)
        
        if __name__ == '__main__':
            
            logging.info(f'{len(tasks)}')

            with ProcessPool(max_tasks = 1) as pool:
                future = pool.map(count_visuals, tasks, timeout=TIMEOUT)                
                logging.info('starting')
                iterator = future.result()
                logging.info('ending...')
                results = []
                while True:
                    try:
                        result = next(iterator)
                        logging.info(f"{result['dcn']} returned.")
                        results.append(result)
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        results.append(None)
                    except ProcessExpired as error:
                        logging.info("%s. Exit code: %d" % (error, error.exitcode))
                    except Exception as error:
                        logging.info("function raised %s" % error)
                        logging.info(error.traceback)  # Python's traceback of remote process
        
        logging.info(f'End of multiprocessing.')
        logging.info(f'{len(results)}')
        x = [r for r in results if r is not None] # filter out 
        
        logging.info(f'Done. Writing results. Number Results: {len(x)}')
        missing_set = set([task['dcn'] for task in tasks]) - set([r['dcn'] for r in x if r is not None])
        logging.info(f'Missing dcn: {missing_set}')
        for item in tasks:
            del item
        tasks.clear()
        del tasks

        df_results = pd.DataFrame(x)
        df_results.to_pickle(os.path.join(RESULTS_FOLDER,'visuals','results',f'visuals_{i}.pickle'))
        
        logging.info('Done.')
        del df_results
        i += 1
        
    result_dfs = []
    logging.info('combining results')
    for f_name in glob(os.path.join(RESULTS_FOLDER,'visuals','results','visuals_*.pickle')):
        result_dfs.append(pd.read_pickle(f_name))
    final_df = pd.concat(result_dfs,axis = 0)
    final_df.to_pickle(os.path.join(RESULTS_FOLDER,'visuals','results','visual.pickle'))
    final_df[['dcn', 'pic_count']].to_stata(os.path.join(RESULTS_FOLDER,'visuals','results','visual.dta'),write_index= False)
    logging.info(f'Rows: {final_df.shape[0]}')
    logging.info(f'{final_df.head()}')