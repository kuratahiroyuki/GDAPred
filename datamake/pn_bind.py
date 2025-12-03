

import pandas as pd
import argparse
import time
import numpy as np

def read_txt(infile):

    df = pd.read_csv(infile, header=None, index_col=None)
    
    return df


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument( '--species_p', type=str,help='variable')
    parser.add_argument( '--species_n', type=str,help='variable')
    parser.add_argument( '--data_path', type=str,help='path')

    species_p = parser.parse_args().species_p
    species_n = parser.parse_args().species_n
    data_path = parser.parse_args().data_path
    
    train_p = data_path +"/train_%s.txt" %(species_p)
    test_p = data_path +"/test_%s.txt" %(species_p) 
    train_n = data_path +"/train_%s.txt" %(species_n) 
    test_n = data_path +"/test_%s.txt" %(species_n) 

    train_p_df = read_txt(train_p)
    test_p_df = read_txt(test_p)
    train_n_df = read_txt(train_n)
    test_n_df = read_txt(test_n)

    # output
    species = species_p.replace("_p","")   
    train_f = data_path +"/train_%s.txt" %(species) 
    test_f = data_path +"/test_%s.txt" %(species) 
    
    pd.concat([train_p_df,train_n_df]).to_csv(train_f, index=None, header=None)
    pd.concat([test_p_df,test_n_df]).to_csv(test_f, index=None, header=None)

  












