#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
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
 
def output_csv_pandas(filename, data):
    data.to_csv(filename, index = None)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--peptide', help='var')
    parser.add_argument('--infile1', help='file')
    parser.add_argument('--infile2', help='file')
    parser.add_argument('--data_path', help='path')

    infile1 = parser.parse_args().infile1
    infile2 = parser.parse_args().infile2
    peptide = parser.parse_args().peptide
    data_path = parser.parse_args().data_path

    df= pd.read_csv(infile1, header=None, names=["seq","label"])
    df_p = df[df["label"]==1].reset_index(drop=True)
    df_n = df[df["label"]==0].reset_index(drop=True)  
    df_p.to_csv(data_path+"/train_%s_p.txt" %peptide, index=None, header=None)
    df_n.to_csv(data_path+"/train_%s_n.txt" %peptide, index=None, header=None)

    df= pd.read_csv(infile2, header=None, names=["seq","label"])
    df_p = df[df["label"]==1].reset_index(drop=True)
    df_n = df[df["label"]==0].reset_index(drop=True)  
    df_p.to_csv(data_path+"/test_%s_p.txt" %peptide, index=None, header=None)
    df_n.to_csv(data_path+"/test_%s_n.txt" %peptide, index=None, header=None)



