#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='term')
    parser.add_argument('--outfig', type=str, help='term')
    parser.add_argument('--data_file', type=str, help='term')
    parser.add_argument('--encode_method_1', type=str, help='term')
    args = parser.parse_args()

    columns_measure= ['Threshold', 'SEN', 'SPE', 'PRE', 'ACC', 'MCC', 'F1', 'AUC', 'AUPRC']
      
    input_dir = args.input_dir
    outfig = args.outfig   
    data_file_item = args.data_file.strip().split() 
    encode_method_1  = args.encode_method_1.strip().split() 
    
    encode_method_item = encode_method_1    
    encode_method_item = [ 'Binary' if name == 'binary' else name for name in encode_method_item]
    encode_method_item = [ name[:5] if 'W2V' in name else name for name in encode_method_item]
    data_name_item = [data_file[4:-5] for data_file in data_file_item]
    data_name_item = [data_file[:-5] for data_file in data_file_item]
    
    encode_dict={}
    for x, y in zip(encode_method_1, encode_method_item):
        encode_dict[x]=y
    #print(encode_dict)
    
    valid_AUC_matrix = pd.DataFrame([], columns= data_name_item)
    valid_ACC_matrix = pd.DataFrame([], columns= data_name_item)
    valid_MCC_matrix = pd.DataFrame([], columns= data_name_item)
       
    test_AUC_matrix = pd.DataFrame([], columns= data_name_item)
    test_ACC_matrix = pd.DataFrame([], columns= data_name_item)
    test_MCC_matrix = pd.DataFrame([], columns= data_name_item)
                
    for data_name, data_file in zip(data_name_item, data_file_item) :
        df = pd.read_excel(input_dir + "/" + data_file, index_col=0).iloc[:, :len(columns_measure)].dropna()

        measure_dict={}
        for x, y in zip(df.columns, columns_measure):
            measure_dict[x]=y
      
        mid = len(df)//2
        df_valid = df.iloc[:mid, :].rename(index=encode_dict).rename(columns=measure_dict)
        df_test  = df.iloc[mid:, :].rename(index=encode_dict).rename(columns=measure_dict)
        #print(df_valid)
        #print(df_test)

        valid_AUC_matrix[data_name] = df_valid['AUC']
        test_AUC_matrix[data_name] = df_test['AUC']
        valid_ACC_matrix[data_name] = df_valid['ACC']
        test_ACC_matrix[data_name] = df_test['ACC']
        valid_MCC_matrix[data_name] = df_valid['MCC']
        test_MCC_matrix[data_name] = df_test['MCC']        

    fig = plt.figure(figsize=(15, 12))
    fig.subplots_adjust(bottom=0.05, left=0.1, top=0.90, right=0.9, wspace=0.2, hspace=0.3)

    ax = fig.add_subplot(2,3,1)
    ax = sns.heatmap(valid_AUC_matrix, annot=True, cmap="Spectral", cbar=False)
    plt.title('AUC Training', fontsize=16 )  #fontweight="bold"
    ax.set_xticklabels(data_name_item, fontsize=12, rotation=60) #, fontweight="semibold")
    ax.set_yticklabels(encode_method_item, fontsize=12)

    ax = fig.add_subplot(2,3,4)
    ax = sns.heatmap(test_AUC_matrix, annot=True, cmap="Spectral", cbar=False) #vmin=-1, vmax=1
    plt.title('AUC Testing', fontsize=16)  
    ax.set_xticklabels(data_name_item, fontsize=12, rotation=60)#
    ax.set_yticklabels(encode_method_item, fontsize=12)

    
    encode_method_item = [ '' for name in encode_method_item]
        
    ax = fig.add_subplot(2,3,2)
    ax = sns.heatmap(valid_MCC_matrix, annot=True, cmap="Spectral", cbar=False)
    plt.title('MCC Training', fontsize=16)  
    ax.set_xticklabels(data_name_item, fontsize=12, rotation=60)
    ax.set_yticklabels(encode_method_item, fontsize=12)
      
    ax = fig.add_subplot(2,3,5)
    ax = sns.heatmap(test_MCC_matrix, annot=True, cmap="Spectral", cbar=False)
    plt.title('MCC Testing', fontsize=16 )  #fontweight="bold"
    ax.set_xticklabels(data_name_item, fontsize=12, rotation=60) #, fontweight="semibold")
    ax.set_yticklabels(encode_method_item, fontsize=12)

    ax = fig.add_subplot(2,3,3)
    ax = sns.heatmap(valid_ACC_matrix, annot=True, cmap="Spectral", cbar=False) #
    plt.title('ACC Training', fontsize=16)  
    ax.set_xticklabels(data_name_item, fontsize=12, rotation=60)#
    ax.set_yticklabels(encode_method_item, fontsize=12)
    
    ax = fig.add_subplot(2,3,6)
    ax = sns.heatmap(test_ACC_matrix, annot=True, cmap="Spectral", cbar=False)
    plt.title('ACC Testing', fontsize=16)  
    ax.set_xticklabels(data_name_item, fontsize=12, rotation=60)
    ax.set_yticklabels(encode_method_item, fontsize=12)
    
    plt.savefig(outfig, dpi=300)
    #plt.show()
