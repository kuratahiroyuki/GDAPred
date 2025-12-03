import sys
import os
import pandas as pd
import argparse
import torch
from torch import autograd
from utils_32 import create_folder, get_conversion_table, read_fasta
from utils_32 import get_encoded_seqs, generate_seqs, get_batch_simple_identity
from utils_32 import plot_loss, plot_identity, write_fasta, write_csv
from model_3 import get_model_and_optimizer_20, get_model_and_optimizer_30, get_model_and_optimizer_40


def calculate_gradient_penalty(discriminator, real_data, fake_data, lambda_value=None):
    lambda_value = lambda_value or 10
    alpha = torch.rand(len(real_data), 1)
    alpha = alpha.expand(real_data.permute(2, 3, 0, 1).size())
    fake_data = fake_data.permute(2, 3, 0, 1)
    real_data = real_data.permute(2, 3, 0, 1)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    interpolates = interpolates.permute(2, 3, 0, 1)
    interpolated_score = discriminator(interpolates)
    grad_outputs = torch.ones(interpolated_score.size())
    gradients = autograd.grad(
        outputs=interpolated_score,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradient_penalty = ((gradients[0].norm(2, dim=1) - 1) ** 2).mean() * lambda_value
    return gradient_penalty


def train(seqs, seqwin, table, batch_size, latent_size, num_epoch, identity_step, gen_interval):
    if seqwin == 20:
        returned = get_model_and_optimizer_20(latent_size, 6, 64)  # encode_num = 6
    elif seqwin == 30:
        returned = get_model_and_optimizer_30(latent_size, 6, 64)  # encode_num = 6    
    elif seqwin == 40:
        returned = get_model_and_optimizer_40(latent_size, 6, 64)  # encode_num = 6
    else:
        print("ERROR, seqwin is out of the definition")
        exit()
        
    generator, discriminator, gen_optim, discrim_optim = returned
    encoded = get_encoded_seqs(seqs, seqwin, table) #(19, 1, 30, 6)  len=30, encode_mum=6
    dataset = torch.utils.data.TensorDataset(torch.Tensor(encoded))
    gen_losses = []
    discrim_losses = []
    identities = [] 
    collected_seqs = {}
    
    print("Starting training loop...")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    
    for epoch in range(1, 1 + num_epoch):
        generated_seqs = {}
        gen_loss_epoch = [] 
        discrim_loss_epoch = [] 
        print("epoch", epoch)
        discriminator.train() 
        generator.train() 
        
        for i, data in enumerate(dataloader):
            discrim_optim.zero_grad() 
                       
            real_seqs = data[0]
            real_disc_val = discriminator(real_seqs).view(-1)
            noise = torch.randn(batch_size, latent_size, 1, 1)
            fake_seqs = generator(noise)
     
            fake_disc_val = discriminator(fake_seqs.detach()).view(-1)
            gp = calculate_gradient_penalty(discriminator, real_seqs, fake_seqs)
            discrim_loss = -torch.mean(real_disc_val) + torch.mean(fake_disc_val) + gp  # gp is very influential.
                      
            #print(f"fake_seqs: {fake_seqs}")
            #print(f"encoded: {encoded.shape}") #(19, 1, 30, 6)  len=30, encode_mum=6
            #print(f"noise: {noise.shape}") # [8, 100, 1, 1]
            #print(f"real_seqs: {real_seqs.shape}") # Torch.Size([8, 1, 30, 6])
            #print(f"fake_seqs: {fake_seqs.shape}") # Torch.Size([8, 1, 30, 6])
            #print(f"fake_disc_val: {fake_disc_val.shape}") # Torch.Size([8])
            #print(f"real_disc_val: {real_disc_val.shape}") # Torch.Size([8])
            #exit()           
               
            discrim_loss.backward()
            discrim_optim.step()
            
            discrim_loss_epoch.append(discrim_loss.item()) ###PyTorchテンソルtorch.Tensorの要素をPython組み込み型（intやfloat）の値として取得するにはitem()メソッドを使う。
                                  
            if i % gen_interval == 0: # generator training default = 5
                gen_optim.zero_grad() 
                fake_disc_val = discriminator(fake_seqs).view(-1)
                gen_loss = -torch.mean(fake_disc_val)
                gen_loss.backward()
                gen_optim.step()               
                gen_loss_epoch.append(gen_loss.item()) ###
                                  
            if epoch % identity_step == 0:
                num_eval = i*batch_size
                
                generator.eval()
                with torch.no_grad():
                    generated_seqs.update( generate_seqs(generator, seqwin, table, noise, num_eval) ) # trained generator is used                  
                    #print(f"seqs {seqs}")
                    #print(f"generated_seqs {len(generated_seqs)}") #8
                    #print(f"generated_seqs {generated_seqs}")  # {'seq_1': 'RWQDMMNDMMNNMMNDMYDDWRERRRRRWR',   }
                    
        if epoch % identity_step == 0:            
            identities.append(get_batch_simple_identity(generated_seqs, seqs))          
            collected_seqs[epoch] = generated_seqs
        
                      
        discrim_losses.append(sum(discrim_loss_epoch)/len(discrim_loss_epoch)) ###
        gen_losses.append(sum(gen_loss_epoch)/len(gen_loss_epoch)) ###
        
    #print(f"gen_losses {gen_losses}")
    #print(f"discrim_losses {discrim_losses}")
    df_loss = pd.DataFrame([], index = [i for i in range(1, num_epoch+1)], columns = ['Dis','Gen'])
    df_loss['Dis'] = discrim_losses
    df_loss['Gen'] = gen_losses
    print(df_loss)
    
    return collected_seqs, identities, gen_losses, discrim_losses, generator, df_loss


def main(peptide, num_gen, csv_path, output_root, seqwin, cutoff, batch_size=None, epoch=None, step=None, gen_interval=5):

    create_folder(output_root)
    batch_size = batch_size or 128
    epoch = epoch or 10000
    step = step or 100
    
    cutoff = cutoff
    
    latent_size = 100
    table_path = os.path.join(os.path.dirname(__file__), "physical_chemical_61.txt")
    table = get_conversion_table(table_path)
    #seqs = read_fasta(fasta_path)
    print(f"{table=}")
    
    df = pd.read_csv(csv_path, header=None)
    seqs = df[df[1]==1].reset_index(drop=True)[0].to_dict()
    
    returned = train(seqs, seqwin, table, batch_size, latent_size, epoch, step, gen_interval)
    
    collected_seqs, identities, gen_losses, discrim_losses, generator, df_loss = returned
    plot_loss(gen_losses, discrim_losses, os.path.join(output_root, "loss_figure_%s_c%s.png" %(peptide, cutoff)))
    if len(identities) > 0:
        path = os.path.join(output_root, "identity_step_%s_c%s.png" %(peptide, cutoff))
        plot_identity(identities, step, path)
    for i, seqs in collected_seqs.items():
        path = os.path.join(output_root, "epoch_{}_generated_seq.fasta".format(i))
        #write_fasta(seqs, path)
        #write_csv(seqs, path)
        
    #torch.save(generator, os.path.join(output_root, "generator.pkl"))
    noise = torch.randn(num_gen, latent_size, 1, 1)
    generated_seqs = generate_seqs(generator, seqwin, table, noise, 0)
    #write_fasta(generated_seqs, os.path.join(output_root, "final_generated_seq.fasta"))
    write_csv(generated_seqs, os.path.join(output_root, "gnc_%s_%s_c%s.txt" %(peptide, peptide, cutoff)))
    df_loss.to_csv(os.path.join(output_root, "loss_change_%s_c%s.csv" %(peptide, cutoff)))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("This program would train the GAN model with user-provided "
                                     "sequences and save the result in thr output folder. "
                                     "Use -h to get more help.")
    parser.add_argument("--peptide",required=True,
                        help="The name of peptide generated")                                     
    parser.add_argument("--num_gen",type=int,
                        help="number of generated peptides = 100")     
    parser.add_argument("--csv_path",required=True,
                        help="The path of the peptide file in CSV format")
    parser.add_argument("--output_root",required=True,
                        help="The path of the folder where result is saved")
    parser.add_argument("--seqwin",type=int,
                        help="sequence window = 20")     
    parser.add_argument("--batch_size",type=int,
                        help="The batch size of the data. The default value is 128.")
    parser.add_argument("--epoch",type=int,
                        help="The epoch of the training process. The default value is 10000.")
    parser.add_argument("--step",type=int,
                        help="The number of epoch to save the temporary result. The default value is 100.")
    parser.add_argument("--gen_interval",type=int,
                        help="Interval of generator training Default value = 5")
    parser.add_argument("--cutoff",type=float,
                        help="Cutoff value for redundant peptides")                        
                    
    args = vars(parser.parse_args())
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    main(**args)
