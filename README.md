# GDA-Pred

# Development environment
conda=23.7.1  
python==3.11.7  
pandas==2.1.4  
numpy==1.26.3  
torch==2.1.2  
scikit-learn==1.3.2  
gensim==4.3.2  
lightgbm==4.1.0  
matplotlib==3.8.2  
seaborn==0.13.1  

# Execution
# 1 Setting directories
Users must keep the structure of the directories  

# 2 Execution of the main program
$sh main.sh  
  
# 3 modules
## 3-1 Dataset construction
/datamake  
const_dataset.sh  
   
 
## 3-2 Peptide generation by GANs
/pepgan  
gan_peptide.sh  

## 3-3 Classification and selection
/classify  
classify_test_fake_12.sh  

## 3-4 Execution of generative data augmentation (GDA)
/classify  
gda.sh  

## 3-5 Visualization of result
/classify
eval_1.sh  

## 3-6 Construction word2vec models
Users must construct the w2v models with Kmer=1,2,3,4 themselves.  
/classify/pepcom/w2v  
$sh general_w2v_const.sh  

# References on peptide encodings and GANs
https://ilearn.erc.monash.edu/  
https://github.com/lsbnb/amp_gan   
https://www.uniprot.org/help/downloads?utm_source=chatgpt.com  

# History
From GDAPred2 in CERVO


