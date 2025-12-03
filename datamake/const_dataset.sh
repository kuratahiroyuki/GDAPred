#!/bin/bash
set -e

peptide=$1
cutoff_list=$2

#peptide=il13
#cutoff_list="0.6 0.7"

cutoff_list_2="${cutoff_list// /,}"
kfold=5
data_path=./${peptide}/dataset
cutoff_path=./${peptide}/data_cutoff

mkdir -p ${data_path}
mkdir -p ${cutoff_path}

infile1=./data/dataset/${peptide}_train.txt  #generating train_${peptide}_p.txt train_${peptide}_n.txt test_${peptide}_p.txt test_${peptide}_n.txt
infile2=./data/dataset/${peptide}_test.txt

python remake_dataset.py --peptide ${peptide} --infile1 ${infile1} --infile2 ${infile2} --data_path ${data_path}  

species_p=${peptide}_p
species_n=${peptide}_n
infile_p=${data_path}/train_${peptide}_p.txt
infile_n=${data_path}/train_${peptide}_n.txt
python remove_redundancy.py --infile ${infile_p} --species ${species_p}  --cutoff_list ${cutoff_list_2} --outpath ${cutoff_path}  --data_type train
python remove_redundancy.py --infile ${infile_n} --species ${species_n}  --cutoff_list ${cutoff_list_2} --outpath ${cutoff_path}  --data_type train

infile_p=${data_path}/test_${peptide}_p.txt
infile_n=${data_path}/test_${peptide}_n.txt
python remove_redundancy.py --infile ${infile_p} --species ${species_p}  --cutoff_list ${cutoff_list_2} --outpath ${cutoff_path}  --data_type test
python remove_redundancy.py --infile ${infile_n} --species ${species_n}  --cutoff_list ${cutoff_list_2} --outpath ${cutoff_path}  --data_type test


for cutoff in ${cutoff_list}
do
echo ${cutoff}
    species_p2=${species_p}_c${cutoff}
    species_n2=${species_n}_c${cutoff}
  
    python pn_bind.py --species_p ${species_p2} --species_n ${species_n2} --data_path ${cutoff_path}
done


