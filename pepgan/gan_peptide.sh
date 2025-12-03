#!/bin/bash
set -e

peptide_list=$1
cutoff_list=$2

epoch=1000
gen_interval=30
num_gen=1000

main_path=$(cd .. && pwd)
pwd
GAN_dir=$(basename "$PWD") 

result_path=${main_path}/${GAN_dir}/data/result/e${epoch}g${gen_interval}
if [ ! -s ${result_path} ]; then
mkdir ${result_path}
fi

for peptide in ${peptide_list}
do
    for cutoff in ${cutoff_list}
    do
        case ${peptide} in  
          AIP)
            seqwin=20    ;;
          il6)
            seqwin=30    ;;
          il13)
            seqwin=20    ;;
          *)
            echo "out of definition"
            ;;
        esac


        echo ${peptide}
        csv_file=${main_path}/datamake/${peptide}/data_cutoff/train_${peptide}_c${cutoff}.txt
        python train_gan.py --peptide ${peptide} --num_gen ${num_gen} --csv_path ${csv_file} --output_root ${result_path} --seqwin ${seqwin} --batch_size 16 --epoch ${epoch} --step 10 --gen_interval ${gen_interval} --cutoff ${cutoff}

    done

done
