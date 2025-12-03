#!/bin/bash
set -e

type=ori  #use original dataset

echo data construction
cd datamake

bash const_dataset.sh AIP "0.6 0.7" #each peptide, cutoff_list
bash const_dataset.sh il6 "0.6 0.7" 
bash const_dataset.sh il13 "0.6 0.7" 

cd ..

echo generation of peptides
cd pepgan

bash gan_peptide.sh "AIP il6 il13" "0.6 0.7" # peptide list, cutoff_list

cd ..

cd classification or selection
echo add the probability to each generated peptide

bash classify_test_fake_12.sh AIP 20 "0.6 0.7" 
bash classify_test_fake_12.sh il6 30 "0.6 0.7"
bash classify_test_fake_12.sh il13 20 "0.6 0.7"  

echo execution of generative data augmentation
# use original dataset
bash gda.sh control  AIP 20 "0.6 0.7" 0.25 0.5 
bash gda.sh augment  AIP 20 "0.6 0.7" 0.25 0.5 
bash gda.sh control il6 30 "0.6 0.7" 0.25 0.5 
bash gda.sh augment il6 30 "0.6 0.7" 0.25 0.5  
bash gda.sh control il13 20 "0.6 0.7" 0.25 0.5 
bash gda.sh augment il13 20 "0.6 0.7" 0.25 0.5 

echo visualization
for cutoff in 0.6 0.7
do
    bash eval_1.sh AIP 20 ${cutoff} ${type}  
    bash eval_1.sh il6 30 ${cutoff} ${type}   
    bash eval_1.sh il13 20 ${cutoff} ${type}   
done
    
cd ..



