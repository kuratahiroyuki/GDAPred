import argparse
import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from utils_31 import create_folder, get_conversion_table, read_fasta
from utils_31 import get_encoded_seqs, generate_seqs_2, generate_seqs_21, get_batch_simple_identity
from utils_31 import plot_loss, plot_identity, write_fasta

torch.set_float32_matmul_precision('medium') 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
timesteps = 100  # Number of time steps  default 200 
beta_start = 1e-4
beta_end = 2e-2

# Linear beta schedule
betas = torch.linspace(beta_start, beta_end, timesteps).to(device)  #betas[time]
alphas = (1.0 - betas).to(device)
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0).to(device)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas) # Standard deviation
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) #alfa(t)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod) # 1-alfa(t)
posterior_variance = (betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    #out = a.gather(-1, t.cpu()) # aの最後の次元 ⇒ timestepに対応するalphaを取ってくる
    out = a.gather(-1, t) 
    return out.reshape(batch_size, *((1,) * (len(x_shape) -  1))) # バッチサイズ x 1 x 1 x 1にreshape

  
def q_sample(x_start, t, noise):
    sqrt_alphas_cumprod_t = extract( sqrt_alphas_cumprod, t, x_start.shape )
    # standard deviation at t
    sqrt_one_minus_alphas_cumprod_t = extract( sqrt_one_minus_alphas_cumprod, t, x_start.shape )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise    


def p_sample(model, x, t, t_index, prompt):
    betas_t = extract(betas, t, x.shape) # beta_t
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape) # 1 - √\bar{α}_t
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)   # 1 / √α_t
    predicted_noise = model(x, t, prompt) 
    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t) # μ_Θ calculation model(x, t)

    if t_index == 0:
        return model_mean, predicted_noise
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape) # σ^2_t calculation
        noise = torch.randn_like(x) # normal random number

    return model_mean + torch.sqrt(posterior_variance_t) * noise, predicted_noise # x_t1
    

class SimpleUNet(nn.Module):
    def __init__(self, seqwin, dim_input, dim_time, dim_prompt):
        super(SimpleUNet, self).__init__()
        self.seqwin = seqwin
        self.dim_input = dim_input
        self.dim_time = dim_time
        self.dim_prompt = dim_prompt
        
        self.down = nn.Sequential(
            nn.Conv1d(dim_input+dim_time+dim_prompt, 64, kernel_size=3, padding=1),  #Batch_size,feature,Length
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, dim_input, kernel_size=3, padding=1)
        )

    def forward(self, x, ts, prompt):  
        t = ts[0].repeat(1, self.seqwin, self.dim_time)
        for i in range(x.shape[0]-1):    
            t = torch.cat((t, ts[i+1].repeat(1, self.seqwin, self.dim_time)), dim=0)
        #print(f"{t.shape=}")
        #print(f"{x.shape=}")         
        x = torch.cat([x] + [t] + [prompt], dim = 2) #BLS  t/timesteps
               
        x = x.permute(0,2,1) #BFL
        x = self.down(x)
        x = self.middle(x)
        x = self.up(x)
        x = x.permute(0,2,1) #[2,4,3]

        return x  
       
     
class DiffusionLightningModel(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x0 = batch[0]
        prompt = batch[1]
        noise = torch.randn_like(batch[0])

        t = torch.randint(0, timesteps, (x0.shape[0],), device=device).long()  # Random time step      
        x_t = q_sample(x0, t, noise)        
        predicted_noise = self.model(x_t, t, prompt) 
        #loss = F.l1_loss(noise, predicted_noise)/x0.shape[0] 
        loss = F.mse_loss(predicted_noise, noise)/x0.shape[0]   
        self.log("train_loss_epoch", loss, on_step=True, on_epoch=True)         
        #print(f"Training Loss: {loss}")
        return loss
        
    def validation_step(self, batch, batch_idx):
        x0 = batch[0]
        prompt = batch[1]
        noise = torch.randn_like(batch[0])
        t = torch.randint(0, timesteps, (x0.shape[0],), device=device).long()  # Random time step      
        x_t = q_sample(x0, t, noise)        
        predicted_noise = self.model(x_t, t, prompt) 
        loss = F.mse_loss(predicted_noise, noise)/x0.shape[0]   
        self.log("validation_loss_epoch", loss, on_step=True, on_epoch=True)         
        return loss        

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def test_step(self, batch, batch_idx):
        x0 = batch[0]
        prompt = batch[1]
        noise = torch.randn_like(x0)
        t = torch.randint(0, timesteps, (x0.shape[0],), device=device).long()  # Random time step      
        x_t = q_sample(x0, t, noise)        
        predicted_noise = self.model(x_t, t, prompt) 
        #loss = F.l1_loss(noise, predicted_noise)/x0.shape[0]
        loss = F.mse_loss(predicted_noise, noise)/x0.shape[0]  
        print(f"Test Loss: {loss}")
        return loss
                 
    def denoise(self, x_t1, prompt, seqwin, dim_input, dim_time, dim_prompt): 
        batch_size = x_t.shape[0]
                 
        x_t1_s = []
        for t in tqdm(reversed(range(0, timesteps)), total=timesteps):
            x_t1_noise = p_sample(self.model.to(device), x_t1.to(device), torch.full((batch_size,), t, device=device, dtype=torch.long), t, prompt.to(device))
            x_t1, predicted_noise = x_t1_noise[0], x_t1_noise[1]
            x_t1_s.append(x_t1.detach().cpu().numpy())
            if t > 0.9*timesteps or t%int(timesteps/10) == 0 or t == 0:  # 
                print(f"predicted_noise {t}: {predicted_noise.pow(2).sum()/batch_size}")
        
        return x_t1_s, x_t1
        
    def forward(self):
        print("Here is forward function")
        return
        
        
class PeptideDataset(Dataset):
    def __init__(self, encoded_seqs, prompt, seqwin, dim_input):
        self.encoded_seqs = encoded_seqs
        self.prompt = prompt
        self.seqwin = seqwin
        self.dim_input = dim_input
        
    def __len__(self):
        return len(self.encoded_seqs)
    
    def __getitem__(self, idx):
        encoded_seq = self.encoded_seqs[idx]["seq"]        
        return torch.tensor(encoded_seq, dtype=torch.float32), self.prompt[self.encoded_seqs[idx]["label"]]


class LossCallback(pl.Callback):
    def __init__(self):
        self.train_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        avg_loss = trainer.callback_metrics["train_loss_epoch"].item()
        self.train_losses.append(avg_loss)   


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--peptide_list', type=str, help='Path')
    parser.add_argument('--target', type=str, help='Path')
    parser.add_argument('--train_file', type=str, help='Path')
    parser.add_argument('--seqwin', type=int, help='Path')
    parser.add_argument('--out_dir', type=str, help='Path')
    parser.add_argument('--num_gen', type=int, help='Path')
    parser.add_argument('--max_epochs', type=int, help='Path')
    parser.add_argument('--cutoff', type=float, help='Path')
                        
    peptide_list = parser.parse_args().peptide_list
    peptide_list = peptide_list.strip().split("_")
    target = parser.parse_args().target
    train_file = parser.parse_args().train_file
    out_dir = parser.parse_args().out_dir
    seqwin = parser.parse_args().seqwin
    num_gen = parser.parse_args().num_gen  
    max_epochs = parser.parse_args().max_epochs #1000
    cutoff = parser.parse_args().cutoff #1000
        
    batch_size = 8
    dim_input = 6
    dim_time = 1   
     
    #peptide_list = ["il13", "AIP", "AMP"] # ["AMP","AIP"] # ["AIP", "ACP"]
    #target = "il13" 
    
    os.makedirs(out_dir, exist_ok=True) 
    dim_prompt = len(peptide_list) 
    
    if len(peptide_list) == 1:
        target = peptide_list[0]
    else:
        target = target  
    
    name = "dmc"
    for peptide in peptide_list:
        name = name  + "_%s" %peptide  
    name = name + "_%s_c%s" %(target, cutoff)    
    loss_file="%s/%s.png" %(out_dir, name)  # diffusion model, physical_chemical_property
    outfile = "%s/%s.txt" %(out_dir, name)
    table_path = "physical_chemical_61.txt"
    table = get_conversion_table(table_path)   
      
    prompt_dict = {}
    for idx, peptide in enumerate(peptide_list):
        prompt_source = torch.zeros([seqwin, dim_prompt], dtype=torch.float32)
        prompt_source[:,idx] = 1
        prompt_dict[peptide] = prompt_source
    
    # prompt for generation
    idx = [i for i, x in enumerate(peptide_list) if x == target]  
    prompt_gen = torch.zeros([seqwin, dim_prompt], dtype=torch.float32)
    #prompt_gen[:int(seqwin/2), 0] = 1
    #prompt_gen[int(seqwin/2):, 1] = 1 
    prompt_gen[:,idx] = 1   
    
       
    # dataset preparation 
    peptide_all = []
    for peptide in peptide_list:       

        df = pd.read_csv(train_file, header=None)
        df = df[df[1]==1]
        df = df.drop(columns=[1])
        df[1] = peptide
        peptide_all.append(df)

    df = pd.concat(peptide_all).reset_index(drop=True)
    print(df)
      
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_peptides, valid_peptides = {}, {}
    for idx in range(train_df.shape[0]):
        train_peptides[idx] = {"seq": train_df.iloc[idx, 0], "label": train_df.iloc[idx, 1]}
    for idx in range(valid_df.shape[0]):
        valid_peptides[idx] = {"seq": valid_df.iloc[idx, 0], "label": valid_df.iloc[idx, 1]}        
    print(train_peptides) 
 
    
    train_encoded = get_encoded_seqs(train_peptides, seqwin, table)   # dict index:{"seq":, "label":}
    valid_encoded = get_encoded_seqs(valid_peptides, seqwin, table)   
    print(train_encoded[1]["seq"]) 
   

    train_peptide_dataset = PeptideDataset(train_encoded, prompt_dict, seqwin, dim_input)
    valid_peptide_dataset = PeptideDataset(valid_encoded, prompt_dict, seqwin, dim_input)
    print(f"peptide_dataset: {train_peptide_dataset}")
    
    # model construction   
    model = SimpleUNet(seqwin, dim_input, dim_time, dim_prompt)
    diffusion_model = DiffusionLightningModel(model)
    train_dataloader = DataLoader(train_peptide_dataset, batch_size=batch_size, shuffle=True, num_workers=12)    
    valid_dataloader = DataLoader(valid_peptide_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss_epoch",
        filename="diffusion-{epoch:02d}-{train_loss:.2f}",
        save_top_k = 1,
        mode="min",
        dirpath="./checkpoints",
    )
        
    early_stopping_callback = EarlyStopping(
        monitor="train_loss_epoch", #val_loss
        patience=20,
        mode="min",
        verbose=True
    ) 
      
    loss_callback = LossCallback()
    
    trainer = pl.Trainer(max_epochs=max_epochs, 
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=20,
        callbacks = [checkpoint_callback, early_stopping_callback, loss_callback]
        )
    trainer.fit(diffusion_model, train_dataloader, valid_dataloader)   
    torch.save(diffusion_model.state_dict(), "diffusion_model.pth")

    # Test
    test_data={}
    for idx in range(num_gen):
        test_data[idx] = {"seq": torch.randn(seqwin, dim_input), "label": target}

    test_data = PeptideDataset(test_data, prompt_dict, seqwin, dim_input) 
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=12)
        
    denoise_model = SimpleUNet(seqwin, dim_input, dim_time, dim_prompt)
    diffusion_model = DiffusionLightningModel(denoise_model)
    diffusion_model.load_state_dict(torch.load("diffusion_model.pth"))
    
    trainer.test(diffusion_model, test_dataloader)
    
    # Plot loss change for epoch   
    data = loss_callback.train_losses
    epoch = [i for i in range(0,len(data))]  # Time from 0 to 10 with 100 points
    plt.figure(figsize=(8, 4))  # Set the figure size
    plt.plot(epoch, data, label="Loss")  # Plot with a label
    plt.title("Loss change")  # Title of the plot
    plt.xlabel("Epoch")  # Label for x-axis
    plt.ylabel("Loss")  # Label for y-axis
    plt.savefig(loss_file, dpi=300)
    
    # generation ###
    prompt_gen = prompt_gen.unsqueeze(0).expand(num_gen, seqwin, dim_prompt)
    x_t = torch.randn(num_gen, seqwin, dim_input)

    seqs  = generate_seqs_21(diffusion_model.to(device), x_t.to(device), prompt_gen.to(device), table, seqwin, dim_input, dim_time, dim_prompt, 0)
                           
    df = pd.DataFrame(list(seqs.items()), columns=["index", "seq"])
    df["label"]=1
    df=df.drop(columns=["index"])
    print(df)

    if not os.path.isfile(outfile):
        # File doesn't exist; write header
        df.to_csv(outfile, mode='w', index=None, header=None)
    else:
        # Append without header
        df.to_csv(outfile, mode='a', index=None, header=None)    
      
