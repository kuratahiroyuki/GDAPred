import numpy as np
import pandas as pd
import argparse
import prettytable as pt
from valid_metrices_p22 import *
#from valid_metrices_p30 import *


def measure_evaluation(score_val, score_test, inpath, val_file, test_file, kfold):
    test_prob_label = []
    threshold = []
    for i in range(kfold):
    
        infile = inpath + '/' + str(i+1) + '/' + val_file
        result = np.loadtxt(infile, delimiter=',', skiprows=1)        
        prob=result[:,1]
        label=result[:,2]
        
        th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, pred_class, prauc_ = eval_metrics(prob, label) 
        valid_matrices = th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, prauc_

        score_val.iloc[i,0]= th_
        score_val.iloc[i,1]= rec_
        score_val.iloc[i,2]= spe_
        score_val.iloc[i,3]= pre_   
        score_val.iloc[i,4]= acc_
        score_val.iloc[i,5]= mcc_
        score_val.iloc[i,6]= f1_ 
        score_val.iloc[i,7]= auc_                     
        score_val.iloc[i,8]= prauc_

        infile = inpath + '/' + str(i+1) + '/' + test_file
        result = np.loadtxt(infile, delimiter=',', skiprows=1)      
        prob=result[:,1]
        label=result[:,2]      
        #print(f"{result.shape=}")
        
        th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, pred_class, prauc_ = th_eval_metrics(th_, prob, label)
        test_matrices = th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, prauc_
            
        score_test.iloc[i,0]= th_
        score_test.iloc[i,1]= rec_
        score_test.iloc[i,2]= spe_
        score_test.iloc[i,3]= pre_   
        score_test.iloc[i,4]= acc_
        score_test.iloc[i,5]= mcc_
        score_test.iloc[i,6]= f1_ 
        score_test.iloc[i,7]= auc_                     
        score_test.iloc[i,8]= prauc_
       
        test_prob_label.append(result[:,1:])
        threshold.append(th_)
        #print_table(print_results(valid_matrices, test_matrices) )
        
    means = score_val.astype(float).mean(axis='index')
    means = pd.DataFrame(np.array(means).reshape(1,-1), index= ['means'], columns=columns_measure)
    score_val = pd.concat([score_val, means])                  

    means = score_test.astype(float).mean(axis='index')  
    means = pd.DataFrame(np.array(means).reshape(1,-1), index= ['means'], columns=columns_measure)
    score_test = pd.concat([score_test, means]) 

    test_prob_label = np.concatenate(test_prob_label, axis=1) 

    prob_sum = 0
    for i in range(kfold):
        prob_sum += test_prob_label[:,2*i] # 0,2,4,6,8
    test_prob = prob_sum/kfold
    threshold = np.mean(threshold)  
    print(f"{test_prob=}")
    print(f"{threshold=}")
    
    df_test_prob_label = pd.DataFrame(np.array([test_prob, test_prob_label[:,1]]).T, columns=["prob", "label"])
    print(f"{df_test_prob_label=}")
    df_test_prob_label = df_test_prob_label[ df_test_prob_label["label"]==1 ] 
    print(f"{df_test_prob_label=}")

    return score_val, score_test, df_test_prob_label, threshold


if __name__=='__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine_method_1', type=str, help='term')
    parser.add_argument('--encode_method_1', type=str, help='term')
    parser.add_argument('--machine_method_2', type=str, help='term')
    parser.add_argument('--encode_method_2', type=str, help='term')
    parser.add_argument('--species', type=str, help='term')
    parser.add_argument('--score_path', type=str, help='term')
    parser.add_argument('--data_path', type=str, help='term') 
    parser.add_argument('--sel_gen_file', type=str, help='term') 
    parser.add_argument('--gen_test_csv', type=str, help='term')              
    args = parser.parse_args()
    """
    machine_method_1 = args.machine_method_1.strip().split(',')
    encode_method_1  = args.encode_method_1.strip().split(',')
    machine_method_2 = args.machine_method_2.strip().split(',')
    encode_method_2  = args.encode_method_2.strip().split(',')
    """
    machine_method_1 = args.machine_method_1.strip().split()
    encode_method_1  = args.encode_method_1.strip().split()
    machine_method_2 = args.machine_method_2.strip().split()
    encode_method_2  = args.encode_method_2.strip().split()
    
    species = args.species
    score_path = args.score_path
    data_path = args.data_path
    sel_gen_file = args.sel_gen_file   
    gen_test_csv = args.gen_test_csv  

    
    kfold=5

    #score_path='.../data/result_%s' %species
    test_file='test_roc.csv' # input
    val_file='val_roc.csv'
    gen_test_file='gen_test_roc.csv'
    
    val_measure ='val_measures.csv' # output
    test_measure ='test_measures.csv'

    index_fold =[i+1 for i in range(kfold)] 
    columns_measure= ['Threshold', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1', 'AUC', 'AUPRC']

    for encode_method in encode_method_1:  
        for deep_method in machine_method_1:             
            inpath = score_path + '/' + deep_method + '/' + encode_method
            outpath= score_path + '/' + deep_method

            score_val  = pd.DataFrame(data=[], index=index_fold, columns=columns_measure)
            score_test = pd.DataFrame(data=[], index=index_fold, columns=columns_measure)

            score_val, score_test, df_test_prob_label, threshold = measure_evaluation(score_val, score_test, inpath, val_file, test_file, kfold)
            
            score_val.to_csv('%s/val_measures.csv' %inpath, header=True, index=True)        
            score_test.to_csv('%s/test_measures.csv' %inpath, header=True, index=True)
                   
            print(score_val)
            print(score_test)  
            #score_test.to_csv('%s/%s_'% (outpath, encode_method) + test_measure, header=True, index=True)
            #score_val.to_csv('%s/%s_'% (outpath, encode_method) + val_measure, header=True, index=True) 
            
            if encode_method == "BLOSUM62":
                gen_test = pd.read_csv(gen_test_csv, index_col=None)    
                print(f"{gen_test=}")           
                gen_prob = []
                for i in range(kfold):
                    df = pd.read_csv(inpath + '/' + str(i+1) + '/' + gen_test_file)
                    gen_prob.append(df["prob"].values.tolist())
                print(f"{len(gen_prob)=}") #5
                mean_prob = list(map(sum, zip(*gen_prob)))          
                print(f"{len(mean_prob)=}") #1000
                gen_test["prob"] = np.array(mean_prob)/kfold
                gen_test.to_csv(sel_gen_file, header=True, index=None)
     

    for encode_method in encode_method_2:  
        for deep_method in machine_method_2:             
            inpath = score_path + '/' + deep_method + '/' + encode_method
            outpath= score_path + '/' + deep_method

            score_val  = pd.DataFrame(data=[], index=index_fold, columns=columns_measure)
            score_test = pd.DataFrame(data=[], index=index_fold, columns=columns_measure)

            score_val, score_test, df_test_prob_label, threshold = measure_evaluation(score_val, score_test, inpath, val_file, test_file, kfold)
            
            score_val.to_csv('%s/val_measures.csv' %inpath, header=True, index=True)        
            score_test.to_csv('%s/test_measures.csv' %inpath, header=True, index=True)
                               
            print(score_val)
            print(score_test)  

            


                    

