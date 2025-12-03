#!/bin/bash
set -e

target=$1
seqwin=$2
cutoff_list=$3

cd ..
main_path=`pwd`
echo ${main_path}
cd classify

peptide_list=il6_il13_AIP # list of all the employed peptides  il13_AIP_AMP

kcv=5
single=True #True False
data_type=train  # data source of the peptide generation: train or test　
species=fake
out_path=${main_path}/classify/result_test_fake_${seqwin}
mkdir -p ${out_path}

gen_prefix=${main_path}/pepgan/data/result/e1000g30/gnc

data_path=${main_path}/classify/data/dataset  # fake peptides with probability (BLOSUM62)
test_fa=${main_path}/classify/data/dataset/independent_test/independent_test.fa
test_csv=${main_path}/classify/data/dataset/independent_test/independent_test.csv
gen_test_fa=${main_path}/classify/data/dataset/gen_independent_test/gen_independent_test.fa
gen_test_csv=${main_path}/classify/data/dataset/gen_independent_test/gen_independent_test.csv
score_path=${main_path}/classify/data/result_${species}
measure_path=${main_path}/classify/data/result_${species}

space=" "
machine_method_1="LGBM"
encode_method_1="AAC DPC CTDC CTDT CTDD CKSAAP GAAC GDPC GTPC BE EAAC AAINDEX BLOSUM62 ZSCALE"
w2v_encode="W2V_1_128_100_40_1 W2V_2_128_100_40_1 W2V_3_128_100_40_1 W2V_4_128_100_40_1"
encode_method_1w=${encode_method_1}${space}${w2v_encode}

machine_method_2=""
encode_method_2="" 
encode_method_2w="" 

for cutoff in ${cutoff_list}
do
    cd ${main_path}/classify
    echo peptide generation
    if [ ${single} = "True" ]; then
     peptide_list=${target}
     echo  ${peptide_list} 
    else
     echo  ${peptide_list}
    fi
           
    cd pepcom
    
    train_file=${main_path}/datamake/${target}/data_cutoff/train_${target}_c${cutoff}.txt 
    test_file=${main_path}/datamake/${target}/data_cutoff/test_${target}_c${cutoff}.txt
    gen_file=${gen_prefix}_${peptide_list}_${target}_c${cutoff}.txt # file of generated peptides
    outfile=${out_path}/${species}_${peptide_list}_${target}_c${cutoff}.xlsx  # evaluation of classification
    
    python train_division_13.py --infile1 ${train_file} --datapath ${data_path} --kfold ${kcv} 
    python test_fasta_2.py --test_file ${test_file} --gen_file ${gen_file} --test_fa ${test_fa} --test_csv ${test_csv} --gen_test_fa ${gen_test_fa} --gen_test_csv ${gen_test_csv}

    cd program
    cd ml   
    echo classification
    sh train_test_3.sh ${seqwin} ${species} ${kcv} "${machine_method_1}" "${encode_method_1w}"  # with W2V
    cd ..
    
    echo evaluation and generative peptide with probability, #./data/dataset/${peptide_list}_${target}_c${cutoff}_gen_test.txt
    
    sel_gen_file=${data_path}/${peptide_list}_${target}_c${cutoff}_gen_test.txt
    python analysis_634.py --machine_method_1 "${machine_method_1}" --encode_method_1 "${encode_method_1w}" --machine_method_2 "${machine_method_2}" --encode_method_2 "${encode_method_2w}" --species ${species}  --score_path ${score_path} --data_path ${data_path} --sel_gen_file ${sel_gen_file} --gen_test_csv ${gen_test_csv} 
    
    echo output
    python csv_xlsx_341.py --machine_method_1 "${machine_method_1}" --encode_method_1 "${encode_method_1w}" --machine_method_2 "${machine_method_2}" --encode_method_2 "${encode_method_2w}" --species ${species} --outfile ${outfile} --measure_path ${measure_path}
    
    cd ..
    cd ..
    
done






    

