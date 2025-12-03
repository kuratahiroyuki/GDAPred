#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse
import os

def import_txt(filename):
    all_data = []
    with open(filename) as f:
        reader = f.readlines()       
        for row in reader:
            all_data.append(row.replace("\n", "").split(','))          
    return pd.DataFrame(all_data, columns = ["seq", "label"])
    
def import_txt_ind(filename):
    all_data = []
    with open(filename) as f:
        reader = f.readlines() 
        if len(reader) <= 1 :
            all_data = []   
        else :
            reader = reader[1:]   
            for row in reader:
                all_data.append(row.replace("\n", "").split(','))  
    df = pd.DataFrame(all_data, columns = ["ind", "seq", "label"]).drop(columns=["ind"]) 
    
    print(f"{len(reader)=}")
    print(f"{df=}")

    
    return df 
 
def output_csv_pandas(filename, data):
    data.to_csv(filename, index = None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile1', help='file')
    parser.add_argument('--infile2', help='file')
    parser.add_argument('--datapath', help='path')
    parser.add_argument('--kfold', type=int, help='value')
    parser.add_argument('--augment_ratio', type=float, help='value')
    parser.add_argument('--threshold', type=float, help='value')
        
    train_file = parser.parse_args().infile1
    gen_test_file = parser.parse_args().infile2
    out_path = parser.parse_args().datapath
    kfold = parser.parse_args().kfold
    augment_ratio = parser.parse_args().augment_ratio
    threshold = parser.parse_args().threshold
    
    train_data = import_txt(train_file)
    num_train = train_data.shape[0]
    num_pos = train_data[train_data["label"]==str(1)].shape[0] ###
    num_neg = num_train - num_pos
    num_add = int(num_pos*augment_ratio)
    
    gen_test_data = pd.read_csv(gen_test_file, index_col=None)
    gen_test_data = gen_test_data[gen_test_data["prob"]>threshold]
    gen_test_data = gen_test_data.drop(columns=["prob"]).reset_index(drop=True)
    num_test = gen_test_data.shape[0] 
    
    print(f"{num_pos=}") #689
    print(f"{num_train=}") #1378
    print(f"{num_test=}") #264
    print(f"{num_add=}") #172
    
        
    if num_add < num_test : #
        training_data = pd.concat([gen_test_data[:num_add], train_data])        
        #print(f"{gen_test_data[:num_add]=}")
        #print(f"{train_data=}")
        #print(f"{training_data['label']=}")
    else:
        training_data = pd.concat([gen_test_data, train_data])

    training_data = training_data.reset_index(drop=True)
    training_data["label"] = training_data["label"].astype(int)
    print(f"{training_data=}")

    
    count=0
    skf = StratifiedKFold(n_splits = kfold, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(training_data, training_data['label']):
        count += 1
        os.makedirs(out_path + "/cross_val/" + str(count), exist_ok = True)
        output_csv_pandas(out_path+ "/cross_val/" + str(count) + "/cv_train_" + str(count) + ".csv", training_data.loc[train_index,:].sample(frac=1).reset_index(drop=True))
        output_csv_pandas(out_path+ "/cross_val/" + str(count) + "/cv_val_" + str(count) + ".csv", training_data.loc[val_index,:].sample(frac=1).reset_index(drop=True))
















