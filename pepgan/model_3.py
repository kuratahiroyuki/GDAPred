from torch import nn
from torch import optim


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class TransposedCBR(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, padding): #100, 64, (5,1), (1,1), (0,0)
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.model(x)


class CRBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, padding, bias=True):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel, stride, padding, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class Generator_20(nn.Module):
    def __init__(self, in_dim, hidden_num, out_dim): #100,64,128
        super().__init__()
        #self.model = nn.Sequential(
        self.model_1 =     TransposedCBR(in_dim, hidden_num * 8, (4, 1), (1, 1), (0, 0))
        self.model_2 =     TransposedCBR(hidden_num * 8, hidden_num * 4, (4, 1), (1, 1), (0, 0))
        self.model_3 =     TransposedCBR(hidden_num * 4, hidden_num * 2, (4, 1), (1, 1), (0, 0))
        self.model_4 =     TransposedCBR(hidden_num * 2, hidden_num, (4, 1), (2, 1), (1, 0))
        self.model_5 =     nn.ConvTranspose2d(hidden_num, 1, (1, out_dim), (1, 1), (0, 0), bias=False)
        self.model_6 =     nn.Tanh()

    def forward(self, x):  
        #print(f"x: {x.shape}") #[8,100,1,1]
        x = self.model_1(x)                                        #(4-4)/1+1=1
        #print(f"x1: {x.shape}") #[8, 512, 4, 1]
        x = self.model_2(x)                                        #(7-4)/1+1=4
        #print(f"x2: {x.shape}") #[8, 256, 7, 1]
        x = self.model_3(x)                                        #(10-4)/1+1=7
        #print(f"x3: {x.shape}") #[8, 128, 10, 1]
        x = self.model_4(x)                                        #(20+2-4)/2+1=10
        #print(f"x4: {x.shape}") #[8, 64, 20, 1]
        x = self.model_5(x)                                        #(20-1)/1+1=20
        #print(f"x5: {x.shape}") #[8, 1, 20, 21]
        x = self.model_6(x)

        return x


class Discriminator_20(nn.Module):
    def __init__(self, in_dim, hidden_num): # 128, 64
        super().__init__()
        #self.model = nn.Sequential(
        self.model_1 =     CRBlock(1, hidden_num, (1, in_dim), (1, 1), (0, 0), bias=False)
        self.model_2 =     CRBlock(hidden_num, hidden_num * 2, (4, 1), (2, 1), (1, 0), bias=False)
        self.model_3 =     CRBlock(hidden_num * 2, hidden_num * 4, (4, 1), (1, 1), (0, 0), bias=False)
        self.model_4 =     CRBlock(hidden_num * 4, hidden_num * 8, (4, 1), (1, 1), (0, 0), bias=False)
        self.model_5 =     nn.Conv2d(hidden_num * 8, 1, (4, 1), (1, 1), (0, 0), bias=False)
        #)

    def forward(self, x):
        #print(f"dis x: {x.shape}") #torch.Size([8, 1, 20, 21])  
        x = self.model_1(x)                                         #(20-1)/1+1=20
        #print(f"dis x1: {x.shape}") #torch.Size([8, 64, 20, 1])  
        x = self.model_2(x)                                         #(20+2-4)/2+1=10
        #print(f"dis x2: {x.shape}") #torch.Size([8, 128, 10, 1]) 
        x = self.model_3(x)                                         #(10-4)/1+1=7
        #print(f"dis x3: {x.shape}") #torch.Size([8, 256, 7, 1])    
        x = self.model_4(x)                                         #(7-4)/1+1=4
        #print(f"dis x4: {x.shape}") #torch.Size([8, 512, 4, 1])
        x = self.model_5(x)                                         #(4-4)/1+1=1
        #print(f"dis x5: {x.shape}") #torch.Size([8, 1, 1, 1])

        return x


def get_model_and_optimizer_20(latent_size, encoded_num, hidden_size):
    generator = Generator_20(latent_size, hidden_size, encoded_num) #100, 64, 128
    generator.apply(weights_init)
    discriminator = Discriminator_20(encoded_num, hidden_size)
    discriminator.apply(weights_init)
    discrim_optim = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9))
    gen_optim = optim.Adam(generator.parameters(), lr=0.0001, betas=(0, 0.9))
    return generator, discriminator, gen_optim, discrim_optim
    

class Generator_30(nn.Module):
    def __init__(self, in_dim, hidden_num, out_dim): #100,64,6
        super().__init__()
        #self.model = nn.Sequential(
        self.model_1 =     TransposedCBR(in_dim, hidden_num * 8, (5, 1), (1, 1), (0, 0))
        self.model_2 =     TransposedCBR(hidden_num * 8, hidden_num * 4, (4, 1), (2, 1), (1, 0))
        self.model_3 =     TransposedCBR(hidden_num * 4, hidden_num * 2, (8, 1), (1, 1), (1, 0))
        self.model_4 =     TransposedCBR(hidden_num * 2, hidden_num, (4, 1), (2, 1), (1, 0))
        self.model_5 =     nn.ConvTranspose2d(hidden_num, 1, (1, out_dim), (1, 1), (0, 0), bias=False)
        self.model_6 =     nn.Tanh()
        #)

    def forward(self, x):  
        #print(f"x: {x.shape}") #8,100,1,1
        x = self.model_1(x)                                     #(5-5)/1+1=1
        #print(f"x1: {x.shape}") #[8, 512, 5, 1]
        x = self.model_2(x)                                     #(10+2-4)/2+1=5
        #print(f"x2: {x.shape}") #[8, 256, 10, 1]
        x = self.model_3(x)                                     #(15+2-8)/1+1=10
        #print(f"x3: {x.shape}") #[8, 128, 15, 1]
        x = self.model_4(x)                                     #(30+2-4)/2+1=15
        #print(f"x4: {x.shape}") #[8, 64, 30, 1]
        x = self.model_5(x)                                     #(30-1)/1+1=30
        #print(f"x5: {x.shape}") #[8, 1, 30, 6]
        x = self.model_6(x)
        return x
        #return self.model(x)


class Discriminator_30(nn.Module):
    def __init__(self, in_dim, hidden_num):
        super().__init__()
        #self.model = nn.Sequential(
        self.model_1 =     CRBlock(1, hidden_num, (1, in_dim), (1, 1), (0, 0), bias=False)  #encoded_num
        self.model_2 =     CRBlock(hidden_num, hidden_num * 2, (4, 1), (2, 1), (1, 0), bias=False)
        self.model_3 =     CRBlock(hidden_num * 2, hidden_num * 4, (8, 1), (1, 1), (1, 0), bias=False)
        self.model_4 =     CRBlock(hidden_num * 4, hidden_num * 8, (4, 1), (2, 1), (1, 0), bias=False)
        self.model_5 =     nn.Conv2d(hidden_num * 8, 1, (5, 1), (1, 1), (0, 0), bias=False)
        #)

    def forward(self, x):
        #print(f"dis x: {x.shape}") #torch.Size([8, 1, 30, 6])
        x = self.model_1(x)                                     #(30-1)/1+1=30, (6-6)/1+1
        #print(f"dis x1: {x.shape}") #torch.Size([8, 64, 30, 1])
        x = self.model_2(x)                                     #(30+2-4)/2+1=15
        #print(f"dis x2: {x.shape}") #torch.Size([8, 128, 15, 1])
        x = self.model_3(x)                                     #(15+2-8)/1+1=10
        #print(f"dis x3: {x.shape}") #torch.Size([8, 256, 10, 1])
        x = self.model_4(x)                                     #(10+2-4)/2+1=5
        #print(f"dis x4: {x.shape}") #torch.Size([8, 512, 5, 1])
        x = self.model_5(x)                                     #(5-5)/1+1=1
        #print(f"dis x5: {x.shape}") #torch.Size([8, 1, 1, 1])
        return x
        #return self.model(x)


def get_model_and_optimizer_30(latent_size, encoded_num, hidden_size):
    generator = Generator_30(latent_size, hidden_size, encoded_num) #100, 64, 6
    generator.apply(weights_init)
    discriminator = Discriminator_30(encoded_num, hidden_size)
    discriminator.apply(weights_init)
    discrim_optim = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9))
    gen_optim = optim.Adam(generator.parameters(), lr=0.0001, betas=(0, 0.9))
    return generator, discriminator, gen_optim, discrim_optim


class Generator_40(nn.Module):
    def __init__(self, in_dim, hidden_num, out_dim): #100,64,6
        super().__init__()
        #self.model = nn.Sequential(
        self.model_1 =     TransposedCBR(in_dim, hidden_num * 8, (5, 1), (1, 1), (0, 0))
        self.model_2 =     TransposedCBR(hidden_num * 8, hidden_num * 4, (4, 1), (2, 1), (1, 0))
        self.model_3 =     TransposedCBR(hidden_num * 4, hidden_num * 2, (4, 1), (2, 1), (1, 0))
        self.model_4 =     TransposedCBR(hidden_num * 2, hidden_num, (4, 1), (2, 1), (1, 0))
        self.model_5 =     nn.ConvTranspose2d(hidden_num, 1, (1, out_dim), (1, 1), (0, 0), bias=False)
        self.model_6 =     nn.Tanh()
        #)

    def forward(self, x):  
        #print(f"x: {x.shape}") #8,100,1,1
        x = self.model_1(x)                                     #(5-5)/1+1=1
        #print(f"x1: {x.shape}") #[8, 512, 5, 1]
        x = self.model_2(x)                                     #(10+2-4)/2+1=5
        #print(f"x2: {x.shape}") #[8, 256, 10, 1]
        x = self.model_3(x)                                     #(15+2-8)/1+1=10
        #print(f"x3: {x.shape}") #[8, 128, 15, 1]
        x = self.model_4(x)                                     #(30+2-4)/2+1=15
        #print(f"x4: {x.shape}") #[8, 64, 30, 1]
        x = self.model_5(x)                                     #(30-1)/1+1=30
        #print(f"x5: {x.shape}") #[8, 1, 30, 6]
        x = self.model_6(x)
        return x
        #return self.model(x)


class Discriminator_40(nn.Module):
    def __init__(self, in_dim, hidden_num):
        super().__init__()
        #self.model = nn.Sequential(
        self.model_1 =     CRBlock(1, hidden_num,                 (1, in_dim), (1, 1), (0, 0), bias=False)  #encoded_num
        self.model_2 =     CRBlock(hidden_num, hidden_num * 2,     (4, 1), (2, 1), (1, 0), bias=False)
        self.model_3 =     CRBlock(hidden_num * 2, hidden_num * 4, (4, 1), (2, 1), (1, 0), bias=False)
        self.model_4 =     CRBlock(hidden_num * 4, hidden_num * 8, (4, 1), (2, 1), (1, 0), bias=False)
        self.model_5 =     nn.Conv2d(hidden_num * 8, 1, (5, 1), (1, 1), (0, 0), bias=False)
        #)

    def forward(self, x):
        #print(f"dis x: {x.shape}") #torch.Size([8, 1, 40, 6])
        x = self.model_1(x)                                     #(40-1)/1+1=40, (6-6)/1+1
        #print(f"dis x1: {x.shape}") #torch.Size([8, 64, 40, 1])
        x = self.model_2(x)                                     #(40+2-4)/2+1=20
        #print(f"dis x2: {x.shape}") #torch.Size([8, 128, 20, 1])
        x = self.model_3(x)                                     #(20+2-4)/2+1=10
        #print(f"dis x3: {x.shape}") #torch.Size([8, 256, 10, 1])
        x = self.model_4(x)                                     #(10+2-4)/2+1=5
        #print(f"dis x4: {x.shape}") #torch.Size([8, 512, 5, 1])
        x = self.model_5(x)                                     #(5-5)/1+1=1
        #print(f"dis x5: {x.shape}") #torch.Size([8, 1, 1, 1])
        return x
        #return self.model(x)


def get_model_and_optimizer_40(latent_size, encoded_num, hidden_size):
    generator = Generator_40(latent_size, hidden_size, encoded_num) #100, 64, 6
    generator.apply(weights_init)
    discriminator = Discriminator_40(encoded_num, hidden_size)
    discriminator.apply(weights_init)
    discrim_optim = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9))
    gen_optim = optim.Adam(generator.parameters(), lr=0.0001, betas=(0, 0.9))

    return generator, discriminator, gen_optim, discrim_optim
        
