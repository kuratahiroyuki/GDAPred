#!/bin/bash
main_path=`pwd`

pep=$1
seqwin=$2
cutoff=$3
type=$4

#condition=test_fake # augment real_fake test_fake
#condition=augment
#condition=control
condition=cont_aug


if [ ${condition} = "augment" ]; then
    data_file="augment_${pep}_${pep}_c${cutoff}.xlsx"
    input_dir=${main_path}/result_${pep}_${seqwin}
    outfig=${input_dir}/gnc_e1000g30_augment_${pep}_${seqwin}_c${cutoff//./}
    echo ${outfig}
    
elif [ ${condition} = "test_fake" ]; then
    data_file=""
    input_dir=${main_path}/result_test_fake_${seqwin}
    outfig=${input_dir}/gnc_e1000g30_test_fake_${pep}_${seqwin} 
    echo ${outfig}
    
elif [ ${condition} = "cont_aug" ]; then
    data_file="control_${pep}_${pep}_c${cutoff}.xlsx augment_${pep}_${pep}_c${cutoff}.xlsx"
    input_dir=${main_path}/result_${pep}_${seqwin}_${type}
    outfig=${input_dir}/gnc_e1000g30_augment_${pep}_${seqwin}_dc${cutoff//./}_gc${cutoff//./}_both
    echo ${outfig}
        
else
    data_file="control_${pep}_${pep}_c${cutoff}.xlsx"
    input_dir=${main_path}/result_${pep}_${seqwin}
    outfig=${input_dir}/${pep}_${seqwin}_c${cutoff//./}
    echo ${outfig}
fi
<<cout
cout


space=" "
machine_method_1="LGBM"
encode_method_1="AAC DPC CKSAAP GAAC GDPC GTPC CTDC CTDT CTDD BE EAAC AAINDEX BLOSUM62 ZSCALE"
w2v_encode="W2V_1_128_100_40_1 W2V_2_128_100_40_1 W2V_3_128_100_40_1 W2V_4_128_100_40_1"
encode_method_1w=${encode_method_1}$space${w2v_encode}

python hmap_1.py  --encode_method_1 "${encode_method_1w}" --input_dir ${input_dir} --data_file "${data_file}" --outfig ${outfig}



