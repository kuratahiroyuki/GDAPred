#!/bin/sh
set -e

peptide_list=$1
cutoff_list=$2

#peptide_list="il6 il13 AIP"
#cutoff_list="0.6 0.7"

max_epochs=10000 #default10000
num_gen=1000   #default1000
single=True

main_path=$(cd .. && pwd)
pwd
VAE_dir=$(basename "$PWD") 


result_path=${main_path}/${VAE_dir}/data/vae
if [ ! -s ${result_path} ]; then
mkdir ${result_path}
fi

for peptide in ${peptide_list}
do
    for cutoff in ${cutoff_list}
    do
        echo ${cutoff}     
       
        case ${peptide} in  
          il6)
            seqwin=30    ;;
          il13)
            seqwin=20    ;;
          AIP)
            seqwin=20    ;;
          AMP)
            seqwin=40    ;; ###
          ACP)
            seqwin=40    ;; ###
          *)
            echo "out of definition"
            ;;
        esac
        
        if [ ${single} = "True" ]; then
            peptide_list_2=${peptide}
        fi
            
        echo ${peptide}
        
        train_file=${main_path}/datamake/${peptide}/data_cutoff/train_${peptide}_c${cutoff}.txt             
        python train_vae_2.py --train_file ${train_file} --out_dir ${result_path} --target ${peptide} --seqwin ${seqwin} --num_gen ${num_gen} --max_epochs ${max_epochs} --cutoff ${cutoff} --batch_size 128
            
    done

done


