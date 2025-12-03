#!/bin/sh

PATH=
infile=${PATH}/uniprot-filtered-reviewed_yes_swiss_plot_all_9_5_565254_0.90.fasta  #Users download the file themselves.
out_dir=./w2v_model

start_time=`date +%s`

for kmer in 1 2 3 4
do
python general_w2v_const.py --kmer ${kmer} --infile ${infile} --out_dir ${out_dir}
done

end_time=`date +%s`
run_time=$((end_time - start_time))
echo calculation time: $run_time



