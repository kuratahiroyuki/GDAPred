#!/bin/bash
set -e

species=$1  #control augment
target=$2 #peptide
seqwin=$3
cutoff=$4
augment_ratio=$5
threshold=$6

cd ..
main_path=`pwd`
echo ${main_path}
cd classify

kfold=5

out_path=${main_path}/classify/result_${target}_${seqwin}_ori
mkdir -p ${out_path}

data_path=${main_path}/classify/data/dataset
test_fasta=${main_path}/classify/data/dataset/independent_test/independent_test.fa
test_csv=${main_path}/classify/data/dataset/independent_test/independent_test.csv
score_path=${main_path}/classify/data/result_${species} # result of peptide classification
measure_path=${main_path}/classify/data/result_${species}
mkdir -p ${score_path}

space=" "
machine_method_1="LGBM" # LGBM XGB RF SVM NB KN LR"
encode_method_1="AAC DPC CTDC CTDT CTDD CKSAAP GAAC GDPC GTPC BE EAAC AAINDEX BLOSUM62 ZSCALE" 
w2v_encode="W2V_1_128_100_40_1 W2V_2_128_100_40_1 W2V_3_128_100_40_1 W2V_4_128_100_40_1"
encode_method_1w=${encode_method_1}${space}${w2v_encode}

machine_method_2=""
encode_method_2="" 
encode_method_2w="" 

<<cout
cout
    

echo ${target}
cd ${main_path}/classify

cd pepcom
train_pep=${target} # training dataset used for classifying target peptides
train_file=${main_path}/datamake/data/dataset/${train_pep}_train.txt # training dataset for classifier
test_file=${main_path}/datamake/data/dataset/${train_pep}_test.txt # training dataset for classifier

sel_gen_file=${main_path}/classify/data/dataset/${target}_${target}_c${cutoff}_gen_test.txt # selected, generated peptide file
      
if [ ${species} = "control" ]; then
 outfile=${out_path}/${species}_${train_pep}_${target}_c${cutoff}.xlsx
 python train_division_14.py --infile1 ${train_file} --datapath ${data_path} --kfold ${kfold}  #control
 echo control
else
 outfile=${out_path}/${species}_${train_pep}_${target}_c${cutoff}.xlsx # evaluation file of target classification
 python train_division_24.py --infile1 ${train_file} --infile2 ${sel_gen_file} --datapath ${data_path} --kfold ${kfold} --augment_ratio ${augment_ratio} --threshold ${threshold}    
 echo data_augmentation 
fi
 
python test_fasta.py --infile1 ${test_file} --outfile1 ${test_fasta} --outfile2 ${test_csv} 

cd program
cd ml   
echo classification
sh train_test_2.sh ${seqwin} ${species} ${kfold} "${machine_method_1}" "${encode_method_1}" #Binary classification It must be set in advance.  W2V1-4
cd ..

echo evaluation
python analysis_622.py --machine_method_1 "${machine_method_1}" --encode_method_1 "${encode_method_1w}" --machine_method_2 "${machine_method_2}" --encode_method_2 "${encode_method_2w}" --species ${species}  --score_path ${score_path}

echo output
python csv_xlsx_341.py --machine_method_1 "${machine_method_1}" --encode_method_1 "${encode_method_1w}" --machine_method_2 "${machine_method_2}" --encode_method_2 "${encode_method_2w}" --species ${species} --outfile ${outfile}  --measure_path ${measure_path}
    
cd ..
cd ..


    

