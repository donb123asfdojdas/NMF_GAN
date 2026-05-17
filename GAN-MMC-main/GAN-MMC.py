import torch
import math
from tqdm.notebook import tqdm_notebook as tqdm
from torch.utils.data import DataLoader,Dataset
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import pandas as pd
from audtorch.metrics.functional import pearsonr
import glob
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
random.seed(0)
class MFS_Dataset(Dataset):
    def __init__(self):
        #################################################################
        # 读取所有以 "H" 开头的 CSV 文件
        file_paths = sorted(glob.glob("data/train_data/H*.csv"))
        # 初始化一个列表来存储每个 CSV 文件的数据
        H=[]
        for file_path in file_paths:
            # 读取 CSV 文件，指定 header=None 来防止将第一行作为标题
            data = pd.read_csv(file_path, header=None).values
            data=(data-np.min(data)+Config.epsilon)/(np.max(data)-np.min(data))
            H.append(data)
        
        # data_list 的形状为 (n, 4, 8)
        self.H_data = torch.tensor(H, dtype=torch.float32)
        print(self.H_data.shape)
        print("--------------------------------------")
        #################################################################
        # 读取所有以 "L" 开头的 CSV 文件
        file_paths = sorted(glob.glob("data/train_data/L*.csv"))
        L = []
        
        for file_path in file_paths:
            data = pd.read_csv(file_path, header=None).values
            data=(data-np.min(data)+Config.epsilon)/(np.max(data)-np.min(data))
            L.append(data)
        #L = (L-np.min(L))/(np.max(L)-np.min(L))
        self.L_data = torch.tensor(L,dtype=torch.float32)
        print(self.L_data.shape)
        print("--------------------------------------")
        #################################################################
        # 读取所有以 "M" 开头的 CSV 文件
        file_paths = sorted(glob.glob("data/train_data/M*.csv"))
        M_all = []
        
        for file_path in file_paths:
            data = pd.read_csv(file_path, header=None).values
            M_all.append(data)
        self.M_data = torch.tensor(M_all)
        print(self.M_data.shape)
        ##################################################################
    def __getitem__(self,index):
        HF = self.H_data[index]
        LF = self.L_data[index]
        M = self.M_data[index]
        return HF, LF, M
    def __len__(self):
        self.num=len(self.M_data)
        #self.num=2
        return self.num
    
class Discriminator(nn.Module):
    random.seed(0)
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # 定义卷积块
        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)
        
        # 定义全连接层块
        def fc_block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)
        
        # 构建模型
        self.model = nn.Sequential(
            conv_block(1, 64),        # Conv: 输入1通道，输出64通道，3x3卷积
            conv_block(64, 128),      # Conv: 输入64通道，输出128通道，3x3卷积
            nn.Flatten(),             # 展平，准备连接全连接层
            fc_block(128 * 4 * 8, 128),  # FC层: 输入128*4*8维，输出128维
            fc_block(128, 64),        # FC层: 输入128维，输出64维
            nn.Linear(64, 1),         # 最后一层，输出1维
            nn.Sigmoid()              # Sigmoid激活函数
        )

    def forward(self, x):
        # 假设输入形状为 (batch_size, 1, 4, 8)
        x = self.model(x)
        return x


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # Dropout概率
        self.dropout_prob = 0.2

        # 定义3D卷积块，使用 (3, 1, 1) 卷积核和新的填充逻辑
        def conv3d_block(in_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), use_bn=True):
            layers = [
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm3d(out_channels) if use_bn else nn.Identity(),
                nn.LeakyReLU(0.2),
                nn.Dropout3d(p=self.dropout_prob)  # 使用类中定义的Dropout概率
            ]
            return nn.Sequential(*layers)

        # 定义上采样块（保持 W 维度不变）
        def upsample3d_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Dropout3d(p=self.dropout_prob)  # 使用类中定义的Dropout概率
            )

        # 编码路径 (Encoder)
        self.enc1 = conv3d_block(1, 64)     # 输入: 1x4x1x8 -> 输出: 64x4x1x8
        self.enc2 = conv3d_block(64, 128)   # 输出: 128x4x1x8
        self.enc3 = conv3d_block(128, 256)  # 输出: 256x4x1x8
        # 解码路径 (Decoder)
        self.dec1 = upsample3d_block(256, 128)   # 输出: 128x4x1x8
        self.dec2 = upsample3d_block(128, 64)    # 输出: 64x4x1x8
        self.final_conv = nn.Conv3d(64, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0))  # 输出: 1x4x1x8

        # 输出激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 输入格式转换
        x = x.view(x.size(0), 1, 4, 8)  # 转换为 (batch_size, channels, depth, height, width)
        x = x.unsqueeze(3)  # 在高度维度 H 上增加一个维度，变为 (batch_size, 1, 4, 1, 8)

        # 2. 编码路径 (3D卷积编码)
        x1 = self.enc1(x)   # 64x4x1x8
        x2 = self.enc2(x1)  # 128x4x1x8
        x3 = self.enc3(x2)  # 256x4x1x8

        # 3. 解码路径 (3D卷积解码)
        d1 = self.dec1(x3) + x2  # 跳跃连接，128x4x1x8
        d2 = self.dec2(d1) + x1  # 跳跃连接，64x4x1x8

        # 4. 最后一层卷积和激活函数
        out = self.final_conv(d2)  # 1x4x1x8
        out = self.sigmoid(out)    # 激活函数

        # 5. 调整输出形状为 (batch_size, 1, 4, 8)
        out = out.squeeze(3)  # 去掉 H 维度
        return out
    

   
class Gan():
    def __init__(self, dataset, train_batch_size, train_number_epochs, Data_size, alpha, beta, hnum):
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.train_number_epochs = train_number_epochs
        self.Data_size = Data_size
        # 其他初始化部分
        self.best_loss = float('inf')  # 初始化为正无穷，用于跟踪最小的 Test_loss
        self.best_model_path = 'best_model/'  # 保存最佳模型的路径
        # Use the CPU or GPU based on Config.use_gpu
        self.device = torch.device('cuda' if Config.use_gpu else 'cpu')
        self.alpha = alpha
        self.beta = beta
        self.hnum = hnum
        if Config.use_gpu:
            self.generator = Generator().cuda()
            self.discriminator = Discriminator().cuda()
        else:
            self.generator = Generator()
            self.discriminator = Discriminator()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), Config.lr_G,weight_decay=1e-4)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), Config.lr_D)
        # 初始化鉴别器的学习率调度器
        self.scheduler_G = ReduceLROnPlateau(self.optimizer_G, mode='min', factor=0.5, patience=20, verbose=True, min_lr=1e-6)
        self.scheduler_D = StepLR(self.optimizer_D, step_size=100, gamma=Config.gammaD)

    def discriminator_loss(self,LF_X, real_X, M):
        G_sample = self.generator(LF_X)
        real_label = Variable(torch.ones(real_X.size(0), 1)).to(self.device)
        fake_label = Variable(torch.zeros(LF_X.size(0), 1)).to(self.device)
        D_prob1 = self.discriminator(real_X * M)
        D_prob2 = self.discriminator(G_sample * M)
        # %% Loss
        criterion = torch.nn.BCELoss()
        D_loss1 = criterion(D_prob1, real_label)
        D_loss2 = criterion(D_prob2, fake_label)
        D_loss = D_loss1 + D_loss2
        return D_loss

    def generator_loss(self, LF_X, real_X, M):
        G_sample = self.generator(LF_X)
        Hat_New_X = LF_X *(1- M) + G_sample * M
        # Discriminator
        D_prob = self.discriminator(G_sample * M)
        # %% Loss
        real_label = Variable(torch.ones(LF_X.size(0), 1)).to(self.device)
        criterion = torch.nn.BCELoss()
        #根据标签来判断其是否精准
        G_loss1 = criterion(D_prob, real_label)
        #根据相似度来判断其结构是否相似
        similarity1 = pearsonr(Hat_New_X.view(LF_X.size()[0], -1), real_X.view(LF_X.size()[0], -1))[:, 0]
        similarity2 = pearsonr((real_X *(1- M)).view(LF_X.size()[0], -1), (LF_X * (1-M)).view(LF_X.size()[0], -1))[:, 0]
        G_loss2 = torch.abs(torch.mean(similarity1 - similarity2))
        #直接基于生成数据和标准数据来判断其是否精准
        MSE_train_loss = torch.mean((M * real_X - M * G_sample) ** 2) / torch.mean(M)
        G_loss = G_loss1 + self.alpha * MSE_train_loss + self.beta * G_loss2

        # %% MSE Performance metric
        #MSE_test_loss = torch.mean(((1 - M) * real_X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
        MSE_test_loss=torch.mean((M*real_X-M*G_sample)**2)/torch.mean(M)
        return G_loss, MSE_train_loss, MSE_test_loss

    def train(self): 
        train_dataloader = DataLoader(self.dataset, batch_size=self.train_batch_size, shuffle=False)
        for epoch in range(0, self.train_number_epochs):
            for i, data in enumerate(train_dataloader, 0):
                X_H, X_L, M = data

                X_H = Variable(X_H.view(X_H.size()[0], *self.Data_size)).to(self.device).float()
                X_L = Variable(X_L.view(X_H.size()[0], *self.Data_size)).to(self.device).float()
                M = Variable(M.view(X_H.size()[0], *self.Data_size)).to(self.device).float()

                # if(epoch>=100):
                self.optimizer_D.zero_grad()
                D_loss_curr = self.discriminator_loss(LF_X=X_L, real_X=X_H, M=M)
                D_loss_curr.backward()
                self.optimizer_D.step()
                
                self.optimizer_G.zero_grad()
                G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = self.generator_loss(LF_X=X_L, real_X=X_H, M=M)
                G_loss_curr.backward()
                self.optimizer_G.step()
            if epoch % 20 == 0:
                self.scheduler_D.step()
                test_loss,origin_loss=test(self.generator)
                print('Epoch: {}'.format(epoch), end='\t')
                print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())), end='\t')
                print('Test_loss:',str(test_loss), end='\t')
                print('origin_loss:',str(origin_loss), end='\t')
                #print('G_loss: {:.4}'.format(np.sqrt(G_loss_curr.item())), end='\t')
                print('D_loss: {:.4}'.format(np.sqrt(D_loss_curr.item())))
                #调整生成器的学习率
                self.scheduler_G.step(test_loss)
                # 如果当前的 Test_loss 小于之前的最小值，则保存模型
                if test_loss < self.best_loss:
                    self.best_loss = test_loss
                    torch.save(self.generator.state_dict(), self.best_model_path+f"best_model epoch={epoch} loss={test_loss}.pth")
                    print(f'Saved best model with Test_loss: {test_loss}')
        # 每个 epoch 结束时，调用调度器更新鉴别器的学习率
            if epoch%20==0 and epoch>=Config.train_number_epochs*0.9:
                torch.save(self.generator.state_dict(), self.best_model_path+f"best_model epoch={epoch} loss={test_loss}.pth")
                print(f'Saved best model with Test_loss: {test_loss}')

def test(generator):
    '''
    device = torch.device('cuda' if Config.use_gpu else 'cpu')
    test_dataloader = DataLoader(self.dataset, batch_size=1)
    test_X_H, test_X_L, test_M = next(iter(test_dataloader))
    max, min = self.dataset.maxmin()
    '''
    device = torch.device('cuda' if Config.use_gpu else 'cpu')
    #-------------------------------------------------------------------#
    file_paths = sorted(glob.glob("data/test_data/L*.csv"))
    L = []
    xmin=0
    xmax=0
    for file_path in file_paths:
        data = pd.read_csv(file_path, header=None).values
        xmin=np.min(data)
        xmax=np.max(data)
        data=(data-xmin+Config.epsilon)/(xmax-xmin)
        L.append(data)
    L_data = torch.tensor(L,dtype=torch.float32).to(device)
    #--------------------------------------------------------------------#
    # 读取所有以 "M" 开头的 CSV 文件
    file_paths = sorted(glob.glob("data/test_data/M*.csv"))
    M_all = []
    
    for file_path in file_paths:
        data = pd.read_csv(file_path, header=None).values
        M_all.append(data)
    M_data = torch.tensor(M_all)
    M_data = M_data.float().unsqueeze(1).to(device)  # 形状变为 [1, 1, 4, 8]
    #-----------------------------------------------------------------#
    file_paths = sorted(glob.glob("data/test_data/H*.csv"))
    # 初始化一个列表来存储每个 CSV 文件的数据
    H=[]
    for file_path in file_paths:
        # 读取 CSV 文件，指定 header=None 来防止将第一行作为标题
        data = pd.read_csv(file_path, header=None).values
        data=(data-np.min(data)+Config.epsilon)/(np.max(data)-np.min(data))
        H.append(data)
    
    # data_list 的形状为 (n, 4, 8)
    H_data = torch.tensor(H, dtype=torch.float32)
    H_data = H_data.float().unsqueeze(1).to(device)  # 形状变为 [1, 1, 4, 8]
    
   # 使用传入的训练好的 generator
    generator.eval()  # 评估模式，防止 dropout 等操作影响结果
    with torch.no_grad():  # 禁用梯度计算
        x = generator(L_data)
    MSE_test_loss = torch.sqrt(torch.mean((M_data * H_data - M_data * x) ** 2) / torch.mean(M_data))
    origin_loss = torch.sqrt(torch.mean((M_data * H_data - M_data * L_data) ** 2) / torch.mean(M_data))
    #output=x*(xmax-xmin)+xmin
    #output=output.detach().cpu().numpy()
    #output=output.reshape(output.shape[2], output.shape[3])
    #np.savetxt('test_data/output.csv',output,delimiter=',')
    return MSE_test_loss.detach().cpu().numpy(),origin_loss.detach().cpu().numpy()
    
   
    
    '''
    test_X_L2 = Variable(test_X_L.view(test_X_L.size()[0], *self.Data_size)).to(device).float()
    get_X_H = self.generator(test_X_L2)
    get_X_H = get_X_H.view(*self.Data_size).cpu()
    imputed_X_H = test_M * test_X_H + (1 - test_M) * get_X_H
    imputed_X_H = imputed_X_H * (max - min) + min
    return imputed_X_H

    '''

class Config():
    #name = 'shubert'20
    #num = 400
    alpha = 100
    beta = 80
    train_batch_size = 256
    train_number_epochs = 1500
    Data_size = (1, 4, 8)
    Data_area = 32
    
    lr_G=0.0015#G网络学习率
    lr_D=0.001#D网络学习系率
    gammaD=0.9#更新D网络学习率
    epsilon = 1e-10
    use_gpu = True  # set it to True to use GPU and False to use CPU

if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    if Config.use_gpu:
        torch.cuda.set_device(0)

    name = 'beale'
    for hnum in range(3, 4):#31
        #M_all = np.loadtxt('.\M\hnum='+ str(round(hnum,3)) + '_H_200.csv', delimiter=",", skiprows=0)
        #print(M_all) 
        #M_all=np.loadtxt('test_data\M.csv', delimiter=",", skiprows=0)
       
        #print(M_all)
        Y = np.zeros((Config.Data_area, 10))
        #M = M_all[:,l].reshape(4,8)
        #print(M)
        dataset=MFS_Dataset()
        #print(dataset)
        
        
        GAN = Gan(dataset, Config.train_batch_size, Config.train_number_epochs, Config.Data_size, Config.alpha, Config.beta, hnum)
        GAN.train()
        '''
        y = test()
        
        y=y.detach().numpy().reshape(20,10)
        
        #通过行列置换生成多样化数据
        dataset = MFS_Missing_Dataset(name, Config.num, Config.p_miss, M)
       
        GAN = Gan(dataset, Config.train_batch_size, Config.train_number_epochs, Config.Data_size, Config.alpha, Config.beta, hnum)
        GAN.train()
        
        y = GAN.test()
        y=y.detach().numpy().reshape(20,10)
        #Y[:, l] = y.detach().numpy().reshape(-1,)
    np.savetxt('./GAN-MMC/'+name+'hnum='+str(round(hnum,3))+'_20-10.csv', y, delimiter=',')
        '''