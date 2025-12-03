
import os
import time
from Bio.Align import PairwiseAligner
import pandas as pd
import numpy as np
import argparse

aligner = PairwiseAligner()
aligner.mode = 'global'
aligner.match_score = 1
aligner.mismatch_score = 0
aligner.open_gap_score = 0
aligner.extend_gap_score = 0


def read_txt(infile):
    df = pd.read_csv(infile, names=["seq", "label"], index_col = None)
    return df

def read_csv(infile):
    df = pd.read_csv(infile, index_col = 0)
    df = df.rename(columns={'match': 'seq', 'Unnamed: 0.1':'target'})
    return df

def sequence_identity(seq1, seq2):
    """Calculate normalized identity between two sequences."""
    #alignment = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True, score_only=False)
    alignment = aligner.align(seq1, seq2)
    score = alignment[0].score
    return score / max(len(seq1), len(seq2))

def remove_redundant_sequences(sequences, cutoff=0.9):
    """Remove redundant sequences based on sequence identity cutoff."""
    non_redundant = []
    for seq in sequences:
        if all(sequence_identity(seq, existing) < cutoff for existing in non_redundant):
            non_redundant.append(seq)
    return non_redundant



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument( '--infile', type=str, help='file')
    parser.add_argument( '--species', type=str,help='variable')
    parser.add_argument( '--data_type', type=str,help='variable')
    parser.add_argument( '--cutoff_list', type=str, help='variable')
    parser.add_argument( '--outpath', type=str, help='variable')

    infile = parser.parse_args().infile
    species = parser.parse_args().species
    data_type = parser.parse_args().data_type
    outpath = parser.parse_args().outpath
    cutoff_list = parser.parse_args().cutoff_list.split(",")
    cutoff_list = [float(x) for x in cutoff_list]

    infasta="temp.fa"
    start=time.time()

    if os.path.isdir(outpath) != True:
        os.makedirs(outpath, exist_ok=True)

    df = read_txt(infile)
    #df = read_csv(infile)

    for peptide in [species]:
        for cutoff in cutoff_list:
            if data_type in "train":
                outfile = "./%s/train_%s_c%s.txt" %(outpath, peptide, cutoff)
            else:
                outfile = "./%s/test_%s_c%s.txt" %(outpath, peptide, cutoff)
            filtered = remove_redundant_sequences(df["seq"].values.tolist(), cutoff) 
            #print(f"non-redundant positive samples:{cutoff=}, {len(filtered)}")
            #print("Non-redundant sequences:")
            #for seq in filtered:
            #    print(seq)
            
            df2 = pd.DataFrame(filtered, columns=["seq"])
            print(f"Peptide: {peptide}, cutoff: {cutoff}, number: {df2.shape[0]}")
            if "_p" in peptide:
                df2["label"] = 1
            else:
                df2["label"] = 0
            df2.to_csv(outfile, index=None, header=None)
        #print(f"Peptide: {peptide}, cutoff: 1.0, number: {df.shape[0]}")

    print(f'elapse: {time.time()-start}')


