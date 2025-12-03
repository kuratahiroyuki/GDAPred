import sys
import os
import pandas as pd
import argparse
import torch
from torch import autograd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset
from math import sqrt
from torch.nn import TransformerDecoderLayer
import matplotlib.pyplot as plt
from transformers.modeling_outputs import BaseModelOutput
import csv

import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

def compute_kernel(x, y):
    x_norm = (x ** 2).sum(1).unsqueeze(1)
    y_norm = (y ** 2).sum(1).unsqueeze(0)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    dist = dist / x.size(1)
    return torch.exp(-dist)

def compute_mmd(z, prior_z):
    xx = compute_kernel(z, z)
    yy = compute_kernel(prior_z, prior_z)
    xy = compute_kernel(z, prior_z)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def loss_function(logits, target_ids, z, beta=1.0): #recon_x=出力次元、x=入力次元、z=潜在変数
    """ β-VAE用の損失関数：再構成誤差 + β * MMD """
    logits = logits.reshape(-1, logits.size(-1))
    target_ids = target_ids.reshape(-1)


    BCE_loss = F.cross_entropy(logits, target_ids, reduction="sum")

    prior_z = torch.randn_like(z, device=z.device)
    MMD = compute_mmd(z.detach(), prior_z)

    return BCE_loss + beta * MMD, BCE_loss, MMD

def write_csv(data, filepath):
  with open(filepath, "w", newline="") as f:
    writer=csv.writer(f)
    for row in data:
      f.write(f"{row},1\n")

class VAE(nn.Module):
    def __init__(self, config, latent_size):
        super().__init__()
        self.fc_mu = nn.Linear(config.hidden_size, latent_size)
        self.fc_logvar = nn.Linear(config.hidden_size, latent_size)
        self.latent_size = latent_size

        self.latent_to_hidden = nn.Linear(latent_size, config.hidden_size)  # z から LSTM 初期状態へ

        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.output_fc = nn.Linear(config.hidden_size, config.vocab_size)  # ロジットを出力
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
    def encode(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, decoder_input_ids):
        # z を LSTM の初期 hidden にする
        h0 = self.latent_to_hidden(z).unsqueeze(0)  # (1, batch, hidden_size)
        c0 = torch.zeros_like(h0)

        # decoder_input_ids は embedding 層で変換が必要 (今回は簡略化のため one-hot としますが通常は embedding を使う)
        # 一例として embedding 使用の場合:
        # embedded = self.embedding(decoder_input_ids)
        embedded = self.embedding(decoder_input_ids)

        lstm_out, _ = self.lstm(embedded, (h0, c0))
        logits = self.output_fc(lstm_out)
        return logits

    def decode_inference(self, z, max_len=55, bos_token_id=0):
        input_ids = torch.full((1, 1), bos_token_id, dtype=torch.long, device=z.device)
        h0 = self.latent_to_hidden(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)

        seq_ids = [bos_token_id]

        for _ in range(max_len):
            embedded = self.embedding(input_ids)
            output, hidden = self.lstm(embedded, hidden)
            logits = self.output_fc(output[:, -1, :])
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)
            seq_ids.append(next_token_id.item())
            input_ids = next_token_id.unsqueeze(1)

        return seq_ids

    def forward(self, x, decoder_input_ids):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, decoder_input_ids)
        return logits, mu, logvar, z


if __name__ == "__main__":
    parser = argparse.ArgumentParser("This program would train the GAN model with user-provided "
                                     "sequences and save the result in thr output folder. "
                                     "Use -h to get more help.")
    parser.add_argument("--target",required=True,
                        help="The name of peptide generated")                                     
    parser.add_argument("--num_gen",type=int,
                        help="number of generated peptides = 100")     
    parser.add_argument("--train_file",required=True,
                        help="The path of the peptide file in CSV format")
    parser.add_argument("--out_dir",required=True,
                        help="The path of the folder where result is saved")
    parser.add_argument("--seqwin",type=int,
                        help="sequence window = 20") #配列の長さ   
    parser.add_argument("--batch_size",type=int,
                        help="The batch size of the data. The default value is 128.")
    parser.add_argument("--max_epochs",type=int,
                        help="The epoch of the training process. The default value is 10000.")
    parser.add_argument("--cutoff",type=float,
                        help="Cutoff value for redundant peptides")                        
      
    args = parser.parse_args()
    
    # Set GPU environment
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    peptide = args.target
    num_samples = args.num_gen
    train_file = args.train_file
    out_dir = args.out_dir
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    seqwin = args.seqwin
    cutoff = args.cutoff

    latent_size=32 #潜在空間の次元
    model_name="yarongef/DistilProtBert" #small version of ProtBert
    tokenizer=AutoTokenizer.from_pretrained(model_name, do_lower_case=False, add_pooling_layer=False, force_download=True) #DistilProtbertモデルで使われたトークンID変換

    df = pd.read_csv(train_file, header=None)
    cc = df[df[1] == 1].reset_index(drop=True)[0].tolist() 
    peptides=[] 
    for c in cc: 
      c_seq=" ".join(c) 
      peptides.append(c_seq)

    tokenized = tokenizer(peptides, padding=True, truncation=True,max_length=seqwin, return_tensors="pt") #パディングをする、seqwin文字以上はカットする、テンソルに変換
    input_ids = tokenized["input_ids"] #トークン化されたsequences
    attention_mask = tokenized["attention_mask"] #パディング部分を無視するためのマスク

    #トークンIDをベクトル変換
    config=AutoConfig.from_pretrained(model_name) #モデルや処理の設定値をまとめて管理するオブジェクト

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") #処理を動かすデバイスを決定
    model = AutoModel.from_pretrained(model_name).to(device) #事前学習済みモデルをロードし、デバイスに転送
    input_ids = input_ids.to(device) #トークンIDもデバイスに載せる
    attention_mask = attention_mask.to(device) #attentionマスクもデバイスに載せる
    outputs = model(input_ids=input_ids, attention_mask=attention_mask) #レイヤー正規化  、Multi-head Attention 、スキップ接続 、レイヤー正規、位置単位順伝播層 、スキップ接続を行ってる
    hidden_state = outputs.last_hidden_state #隠れ状態を得る
    h = hidden_state[:, 0].to(device) ##各シーケンスの先頭トークン（例: [CLS] トークン）の埋め込みを取り出している。[CLS]トークンは文の特徴を表す。

    # VAEモデル定義
    model = VAE(config, latent_size).to(device)  # モデルのインスタンス化
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  #  PyTorch の Adam 最適化アルゴリズムを作成し、学習対象のパラメータと学習率を指定. model.parameters() 学習させたいモデルのパラメータ（重み・バイアス）をまとめて渡す。後で optimizer.step() すると、このパラメータたちが更新される。

    loss_list = []
    bce_list = []
    mmd_list = []
    
    min_loss = float("inf")

    decoder_input_ids = input_ids[:, :-1]  # デコーダへの入力（<CLS> など先頭トークン〜最後の1つ前まで）
    target_ids = input_ids[:, 1:]          # 出力のターゲット（1つ後のトークン〜最後まで）

    for epoch in range(1, max_epochs):
        logits, mu, logvar, z = model(h, decoder_input_ids)
        loss, CE, MMD = loss_function(logits, target_ids, z, beta=0.8)

        optimizer.zero_grad()
        grads = torch.autograd.grad(
            loss,
            model.parameters(),
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )
        for param, grad in zip(model.parameters(), grads):
          param.grad = grad
        optimizer.step()

        # ログ表示など
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, CE: {CE.item():.4f}, MMD: {MMD.item():.4f}")
        
        # Early stop    
        if min_loss > loss.item() :
            no_improve = 0
            min_loss = loss.item()        
            loss_list.append(loss.item())
            bce_list.append(CE.item())
            mmd_list.append(MMD.item()) 
            torch.save(model.state_dict(), "%s/model.pth"%(out_dir)) ###
                
        elif epoch > 1000 and loss.item() > loss_list[-1] * 10: # A spike appears in loss function  
            print("A spike emerges and optimiation ends")  
            loss_list.append(loss_list[-1])
            bce_list.append(bce_list[-1])
            mmd_list.append(mmd_list[-1])
            break
                          
        else :
            loss_list.append(loss.item())
            bce_list.append(CE.item())
            mmd_list.append(MMD.item())         
            no_improve += 1
            if no_improve > 20 and loss.item() < 1.0 :               
                break
                                         
    df_loss = pd.DataFrame([], index = [i for i in range(1, epoch+1)], columns = ['LOSS','BCE','MMD'])
    df_loss['LOSS'] = loss_list
    df_loss['BCE'] = bce_list
    df_loss['MMD'] = mmd_list  
    print(df_loss)

    plt.plot(range(1, len(loss_list)+1), loss_list, label="Total_loss")
    plt.plot(range(1, len(bce_list)+1), bce_list, label="BCE")
    plt.plot(range(1, len(mmd_list)+1), mmd_list, label="MMD")
    
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.ylim([0,10])
    #plt.show()
    loss_fig="%s/loss_%s_c%s.png" %(out_dir, peptide, cutoff)
    loss_file="%s/loss_%s_c%s.txt" %(out_dir, peptide, cutoff)
    df_loss.to_csv(loss_file, index=None)
    plt.savefig(loss_fig, dpi=300)
     
    # --------------------------
    # 生成フェーズ（IL-13ペプチドに特化して新しい配列を生成）
    # --------------------------
    model = VAE(config, latent_size).to(device) 
    #model.load_state_dict(torch.load("%s/model_%s_c%s.pth"%(out_dir, peptide, cutoff)))
    model.load_state_dict(torch.load("%s/model.pth"%out_dir))
    model.eval()

    latent_size = model.latent_size
    device = next(model.parameters()).device
    BOS_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0

    generated_seqs = []  # デコード済みの文字列を格納するリスト

    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, latent_size).to(device)
            seq_ids = model.decode_inference(z, max_len=55, bos_token_id=BOS_token_id)
            generated_seq = tokenizer.decode(seq_ids, skip_special_tokens=True)
            generated_seqs.append(generated_seq)  # リストに追加
        generated_seqs = [seq.replace(" ", "") for seq in generated_seqs if len(seq.replace(" ", "")) >= 3]

    # 許容するアミノ酸（標準20種 + パディングX）
    standard_aas = set("ACDEFGHIKLMNPQRSTVWY")

    # 生成シーケンスの後処理
    filtered_seqs = []
    for seq in generated_seqs:
        # 空白を削除して大文字化
        seq_clean = seq.replace(" ", "").upper()
        # 3文字以上かつ標準アミノ酸のみなら残す
        if len(seq_clean) >= 3 and set(seq_clean).issubset(standard_aas):
            filtered_seqs.append(seq_clean)

    generated_seqs = filtered_seqs

    write_csv(generated_seqs, os.path.join(out_dir, "vae_%s_%s_c%s.txt" %(peptide, peptide, cutoff))) ###




