#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import sys
import os
import argparse
import pandas as pd
import Bio
from Bio import SeqIO
import json
from gensim.models import word2vec
import logging
#sys.path.append("/Users/kurata/Documents/research/DeepHV/program/w2v")
#import remove_redundant as rr
import time

normal_amino_asids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]

def file_input_csv(filename,index_col = None):
    data = pd.read_csv(filename, index_col = index_col)
    
    return data

def import_fasta(filename):
    seq_dict = []
    
    for record in SeqIO.parse(filename, "fasta"):
        temp = str(record.seq)
        if(sum([0 if temp[i] in normal_amino_asids else 1 for i in range(len(temp))]) == 0):
            seq_dict.append(temp)
        
    return seq_dict

def kmer_seq_const(seq, num):
    emb_all = []
    for i in range(len(seq)):
        emb_seq = []
        for j in range(len(seq[i]) - num + 1):
            emb_seq.append(seq[i][j:j+num])
        emb_all.append(emb_seq)
        
    return emb_all

################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--kmer', type=int, help='value')
parser.add_argument('--infile', help='value')
parser.add_argument('--out_dir', help='value')

start=time.time()

kmer = parser.parse_args().kmer
input_file = parser.parse_args().infile
path_out = parser.parse_args().out_dir

feature_size = 128 #128
epochs=100 #4
window=40
min_count=1
sg=1

#os.chdir(path_data)
seq_list = import_fasta(input_file)

for i in range(1):  
   kmer_sequences = kmer_seq_const(seq_list, kmer)
   print(kmer_sequences)
   logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
   model = word2vec.Word2Vec(kmer_sequences, vector_size  = feature_size, min_count = min_count, window = window, epochs = epochs, sg = sg)

   os.makedirs(path_out, exist_ok = True)
   model.save(path_out + "/W2V_general_%s_%s_%s_%s_%s.pt" %(kmer, feature_size, epochs, window, sg))

print('elapsed time =', time.time()-start)


