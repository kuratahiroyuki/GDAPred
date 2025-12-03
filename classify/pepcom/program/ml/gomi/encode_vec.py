# from ml_av2.py
import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from gensim.models import word2vec
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score,  mean_squared_error, mean_absolute_error
from encodingAA_23 import AAC, DPC, TPC, PAAC, EAAC, CKSAAP, CKSAAGP, GAAC, GDPC, GTPC, CTDC, CTDT,CTDD, CTriad, AAINDEX, BLOSUM62, ZSCALE

# Check the dimension of encoding vectors.

def aa_dict_construction():
   #AA = 'ARNDCQEGHILKMFPSTWYV'
   AA = 'ARNDCQEGHILKMFPSTWYV'
   keys=[]
   vectors=[]
   for i, key in enumerate(AA) :
      base=np.zeros(21)
      keys.append(key)
      base[i]=1
      vectors.append(base)
   aa_dict = dict(zip(keys, vectors))
   aa_dict["X"] = np.zeros(21)
   return aa_dict
    
#dataset reading
def pad_input_csv(filename, seqwin, index_col = None):
    df1 = pd.read_csv(filename, delimiter=',',index_col = index_col)
    seq = df1.loc[:,'seq'].tolist()
    #data triming and padding
    for i in range(len(seq)):
       if len(seq) > seqwin:
         seq[i]=seq[i][0:seqwin]
       seq[i] = seq[i].ljust(seqwin, 'X')
    for i in range(len(seq)):
       df1.loc[i,'seq'] = seq[i]   
    return df1
      
def pickle_save(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def pickle_read(path):
    with open(path, "rb") as f:
        res = pickle.load(f)      
    return res
    
def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data    


#############################################################################################
if __name__=="__main__":
    
    path="/home/kurata/myproject/pa3/il13_2/data/dataset/cross_val"
    test_file="/home/kurata/myproject/pa3/il13_2/data/dataset/independent_test/independent_test.csv"
    i = 1
    seqwin=25
    train_dataset = pad_input_csv(path + "/" + str(i) + "/cv_train_" + str(i) + ".csv", seqwin, index_col = None)
    val_dataset = pad_input_csv(path + "/" + str(i) + "/cv_val_" + str(i) + ".csv", seqwin, index_col = None)
    test_dataset = pad_input_csv(test_file, seqwin, index_col = None)

    train_seq = train_dataset['seq'].tolist()
    val_seq = val_dataset['seq'].tolist()
    test_seq = test_dataset['seq'].tolist()

    myOrder = 'ARNDCQEGHILKMFPSTWYV' #'ACDEFGHIKLMNPQRSTVWY'
    kw = {'order': myOrder, 'type': 'Protein'}
    train_seq=train_dataset.values.tolist()
    val_seq  =val_dataset.values.tolist()
    test_seq =test_dataset.values.tolist()
 
    for encode_method in ["AAC","DPC","PAAC","CKSAAP","CTriad","GAAC","GDPC","GTPC","CTDC","CTDT","CTDD","EAAC","BLOSUM62","AAINDEX","ZSCALE"]:  
        if encode_method == 'AAC':
            train_X = np.array(AAC(train_seq, **kw), dtype=float)
            valid_X = np.array(AAC(val_seq, **kw), dtype=float)
            test_X = np.array(AAC(test_seq, **kw), dtype=float)
            
        elif encode_method == 'TPC':
            train_X = np.array(TPC(train_seq, **kw), dtype=float)
            valid_X = np.array(TPC(val_seq, **kw), dtype=float)
            test_X = np.array(TPC(test_seq, **kw), dtype=float)

        elif encode_method == 'DPC':
            train_X = np.array(DPC(train_seq, **kw), dtype=float)
            valid_X = np.array(DPC(val_seq, **kw), dtype=float)
            test_X = np.array(DPC(test_seq, **kw), dtype=float)

        elif encode_method == 'EAAC':
            train_X = np.array(EAAC(train_seq,  **kw), dtype=float)
            valid_X = np.array(EAAC(val_seq, **kw), dtype=float)
            test_X = np.array(EAAC(test_seq, **kw), dtype=float)
            
        elif encode_method == 'CTriad':
            train_X = np.array(CTriad(train_seq, **kw), dtype=float)
            valid_X = np.array(CTriad(val_seq, **kw), dtype=float)
            test_X = np.array(CTriad(test_seq, **kw), dtype=float)

        elif encode_method == 'GAAC':
            train_X = np.array(GAAC(train_seq, **kw), dtype=float)
            valid_X = np.array(GAAC(val_seq, **kw), dtype=float)
            test_X = np.array(GAAC(test_seq, **kw), dtype=float)

        elif encode_method == 'GDPC':
            train_X = np.array(GDPC(train_seq, **kw), dtype=float)
            valid_X = np.array(GDPC(val_seq, **kw), dtype=float)
            test_X = np.array(GDPC(test_seq, **kw), dtype=float)

        elif encode_method == 'GTPC':
            train_X = np.array(GTPC(train_seq, **kw), dtype=float)
            valid_X = np.array(GTPC(val_seq, **kw), dtype=float)
            test_X = np.array(GTPC(test_seq, **kw), dtype=float)

        elif encode_method == 'CTDC':
            train_X = np.array(CTDC(train_seq, **kw), dtype=float)
            valid_X = np.array(CTDC(val_seq, **kw), dtype=float)
            test_X = np.array(CTDC(test_seq, **kw), dtype=float)
            
        elif encode_method == 'CTDD':
            train_X = np.array(CTDD(train_seq, **kw), dtype=float)
            valid_X = np.array(CTDD(val_seq, **kw), dtype=float)
            test_X = np.array(CTDD(test_seq, **kw), dtype=float)

        elif encode_method == 'CTDT':
            train_X = np.array(CTDT(train_seq, **kw), dtype=float)
            valid_X = np.array(CTDT(val_seq, **kw), dtype=float)
            test_X = np.array(CTDT(test_seq, **kw), dtype=float)
            
        elif encode_method == 'PAAC':
            train_X = np.array(PAAC(train_seq, **kw), dtype=float)
            valid_X = np.array(PAAC(val_seq, **kw), dtype=float)
            test_X = np.array(PAAC(test_seq, **kw), dtype=float)      

        elif encode_method == 'CKSAAP':
            train_X = np.array(CKSAAP(train_seq, **kw), dtype=float)
            valid_X = np.array(CKSAAP(val_seq, **kw), dtype=float)
            test_X = np.array(CKSAAP(test_seq, **kw), dtype=float)

        elif encode_method == 'CKSAAGP':
            train_X = np.array(CKSAAGP(train_seq, **kw), dtype=float)
            valid_X = np.array(CKSAAGP(val_seq, **kw), dtype=float)
            test_X = np.array(CKSAAGP(test_seq, **kw), dtype=float)
            
        elif encode_method == 'AAINDEX':
            train_X = np.array(AAINDEX(train_seq, **kw), dtype=float)
            valid_X = np.array(AAINDEX(val_seq, **kw), dtype=float)
            test_X = np.array(AAINDEX(test_seq, **kw), dtype=float)

        elif encode_method == 'BLOSUM62':
            train_X = np.array(BLOSUM62(train_seq, **kw), dtype=float)
            valid_X = np.array(BLOSUM62(val_seq, **kw), dtype=float)
            test_X = np.array(BLOSUM62(test_seq, **kw), dtype=float)

        elif encode_method == 'ZSCALE':
            train_X = np.array(ZSCALE(train_seq, **kw), dtype=float)
            valid_X = np.array(ZSCALE(val_seq, **kw), dtype=float)
            test_X = np.array(ZSCALE(test_seq, **kw), dtype=float)  

        else :
            pass
            print('no encode method')
            exit()
        print(f'{encode_method} {train_X.shape}')
 

