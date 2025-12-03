#!/bin/sh


infile1=..../datamake/data/dataset/il6_train.txt
infile2=..../datamake/data/dataset/l6_test.txt
w2v_path=./w2v_model

seqwin=25
kmer=4
size=100 
epochs=16 
sg=1
window=40 

for kmer in  1 2 3 4 
do
python specific_w2v_const.py --infile1 ${infile1} --infile2 ${infile2} --w2v ${w2v_path} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window}
done

      
