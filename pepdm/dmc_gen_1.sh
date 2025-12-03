#!/bin/sh
set -e

peptide_list=$1
cutoff_list=$2

max_epochs=1000 #default1000
num_gen=1000    #default1000
single=True

main_path=$(cd .. && pwd)
pwd
DM_dir=$(basename "$PWD")  #pepdm_1

result_path=${main_path}/${DM_dir}/data/e${max_epochs}
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
            seqwin=40    ;;
          ACP)
            seqwin=40    ;;
          *)
            echo "out of definition"
            ;;
        esac
        
        if [ ${single} = "True" ]; then
            peptide_list_2=${peptide}
        fi
         
        train_file=${main_path}/datamake/${peptide}/data_cutoff/train_${peptide}_c${cutoff}.txt
        python pepdmc_2.py --peptide_list ${peptide_list_2} --train_file ${train_file} --out_dir ${result_path} --target ${peptide} --seqwin ${seqwin} --num_gen ${num_gen} --max_epochs ${max_epochs} --cutoff ${cutoff}

    done

done


