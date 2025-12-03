#!/bin/bash

#set -e
start_time=`date +%s`

cd ..
cd ..
cd ..
main_path=`pwd`
echo ${main_path}
cd pepcom
cd program
cd ml
seqwin=$1
species=$2
kfold=$3
machine_method_1=$4
encode_method_1w=$5

train_path=${main_path}/data/dataset/cross_val
test_csv=${main_path}/data/dataset/independent_test/independent_test.csv
result_path=${main_path}/data/result_${species}
esm2_dict=/home/kurata/myproject/common/esm2_enc/il13_seq2esm2_dict.pkl


for machine_method in ${machine_method_1} #LGBM XGB RF NB KN SVM LR
do
    for encode_method in ${encode_method_1w} #ESM2 # AAC DPC PAAC CKSAAP GAAC GDPC GTPC CTDC CTDT CTDD CTriad BE EAAC BLOSUM62 AAINDEX ZSCALE ESM2  #TPC (too long time)
    do
            echo ${encode_method}
           # exit 1
        #if [[ "${encode_method}" == *"W2V"* ]]; then
        if echo "${encode_method}" | grep -q "W2V"; then
            #kmer=${encode_method:4:1} 
            kmer=$(echo "${encode_method}" | cut -c5)
            encode_method=W2V
            size=128
            epochs=100
            window=40
            sg=1                                                                            
            w2v_model=/home/kurata/myproject/common/w2v_model/w2v_g/W2V_general_${kmer}_128_100_40_1.pt  ###      
            echo ${machine_method} ${encode_method} ${kmer}
            python ml_train_test_46.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --machine ${machine_method}  --encode ${encode_method} --kfold ${kfold} --seqwin ${seqwin} --kmer ${kmer} --w2vmodel ${w2v_model} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window} --esm2 ${esm2_dict}        
        
        else
            w2v_model=None
            kmer=1
            size=-1
            epochs=-1
            window=-1
            sg=-1
            echo ${machine_method} ${encode_method}
            python ml_train_test_46.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --machine ${machine_method}  --encode ${encode_method} --kfold ${kfold} --seqwin ${seqwin} --kmer ${kmer} --w2vmodel ${w2v_model} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window} --esm2 ${esm2_dict}
            
        fi
        
    done
done

end_time=`date +%s`
echo elapsed time: $((end_time-start_time))



