
Word2vec model construction

General W2V model construction (recommendation)
bash general_w2v_const.sh

Dataset-specific W2V model construction
bash specific_w2v_const.sh


Note
W2V require lots of memory.
Kmer=1
W2V_general_1_128_100_40_1.pt  26.3 KB
Kmer=2
W2V_general_2_128_100_40_1.pt  42.4 KB
Kmer=3
W2V_general_3_128_100_40_1.pt  8.4 MB
Kmer=4
W2V_general_4_128_100_40_1.pt + W2V_general_4_128_100_40_1.pt.syn1neg.npy +  W2V_general_4_128_100_40_1.pt.wv.vectors.npy
                 4.5 + 81.9 + 81.9 MB

