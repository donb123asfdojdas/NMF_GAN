import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy import linalg as LA
import numpy as np
import random
import pandas as pd
import math
import scipy.io as sio
# In[2]:


mat = loadmat('singal_test.mat', struct_as_record=False, squeeze_me=True)
data, gnd = mat['fea'].astype('float32'), mat['gnd']


# In[3]:


#高斯噪声
def wgn(x, SNR):
    noise_m = np.random.randn(1024)
    noise_power = np.sum(x) / (10**(SNR/10)) * (noise_m**2) / np.linalg.norm(noise_m)**2    
    noise_power = np.array(noise_power).reshape((1024, 1))
    signal_add_noise = x + noise_power
    return signal_add_noise
#算水平距离
def obj2(a,b):
    L = (a[0]-b[0]) ** 2 +  (a[1]-b[1]) **2 
    return L
#随机撒点生成信号的位置
def generate_random_numbers():
    result = []

    while len(result) < 8:
        random_num = random.randint(1, 100)
        result.append(random_num)
    return result
#随机生成信号的发射功率
def generate_random_p():
    result = []

    while len(result) < 8:
        random_num = random.randint(40, 90)
        result.append(random_num)
    
    return result

#第一层的nmf
def norm(X, p):
    if p == 1:
        tmp = np.abs(X)
        val = np.sum(np.sum(tmp))
    elif p == 2:
        tmp = X * X
        tmp = np.sum(np.sum(tmp))
        val = np.sqrt(tmp)
    else:
        a = 0
        b = 1
        assert a == b, 'The input parameter is incorrect !'
    return val
def nmf(X, r, nIter):
    xRow,xCol = np.shape(X)
    W = Bs.T
   # W = np.random.rand(xRow, r)
  #  W = justNorm(W)
  ################为了保证对比数据和第一次生成数据相同########################
    random.seed(15)
  ########################################
    H = np.random.rand(r, xCol)
 #   H = justNorm(H)
    for ii in range(nIter):
        tmp = np.dot(np.transpose(W), X)
        tnp = np.dot(np.transpose(W), W)
        tnp = np.dot(tnp, H)
        tm = tmp / tnp
        H = H * tm
        tmp = np.dot(X, np.transpose(H))
        tnp = np.dot(W, H)
        tnp = np.dot(tnp, np.transpose(H))
        tm = tmp / tnp
     #   W = W * tm
     #   tmp = np.dot(W, H)
       # obj = X - tmp
       # obj = norm(obj, 1) 
    #    str = 'The %d-th iteration: ' %ii + '%f' %obj
    #    print(str)
    # 将元组转换为列表，方便操作
    H_list = list(H)
    
    # 将四个数组水平堆叠成一个8x4的矩阵，然后转置得到4x8的矩阵
    result = np.hstack(H_list).T

    return W, H
def new_wgn(x, noise_power):
    signal_add_noise = x + noise_power
    return signal_add_noise
def fix_noise(x, SNR):
    noise_m = np.random.randn(1024)
    noise_power = np.sum(x) / (10**(SNR/10)) * (noise_m**2) / np.linalg.norm(noise_m)**2    
    noise_power = np.array(noise_power).reshape((1024, 1))
    return noise_power
def NOISE_POS(id):
    toty = id // 10
    totx = id % 10
    if id % 10 == 0:
        toty = toty - 1
        totx = 10
    y =  8 * toty 
    x =  8 * (totx - 1)
    return x, y
def NOISE_len2(id1,id2,hh):
    a = NOISE_POS(id1)
    b = NOISE_POS(id2)
    L = (a[0]-b[0]) ** 2 +  (a[1]-b[1]) **2 + hh ** 2
    return L 


# In[17]:


#把网格转化成坐标
def POS(id):
    toty = id // 10
    totx = id % 10
    if id % 10 == 0:
        toty = toty - 1
        totx = 10
    y =  8 * toty 
    x =  8 * (totx - 1)
    return x, y
def len2(id1,id2,hh):
    a = POS(id1)
    b = POS(id2)
    L = (a[0]-b[0]) ** 2 +  (a[1]-b[1]) **2 + hh ** 2
    return L 


# In[19]:


#print(POS(1))
#print(POS(2))


# In[5]:
s1 = np.array(data)[0]
s2 = np.array(data)[1]
s3 = np.array(data)[2]
s4 = np.array(data)[3]
s5 = np.array(data)[4]
s6 = np.array(data)[5]
s7 = np.array(data)[6]
s8 = np.array(data)[7]
#初始化固定Bs
bases = []
bases.append(s1)
bases.append(s2)
bases.append(s3)
bases.append(s4)
bases.append(s5)
bases.append(s6)
bases.append(s7)
bases.append(s8)

Bs = np.array(bases)
#print(Bs.shape)

#idm1 = 34
#idm2 = 37
#idm3 = 64
#idm4 = 67




def get_nmf_obj(X, W, H):
    W = np.matrix(W)
    H = np.matrix(H)
    res = LA.norm(X - H * W, 'fro')**2 
    return res
def get_fanshu_obj(X, W, H, rho):
        W = np.matrix(W)
        H = np.matrix(H)
        '''
        objective function
            1/2 ||X - WH||_{F}^{2} + 0.5 * rho * sum{||hj||_{1}^{2} - ||hj||_{2}^{2}} + 0.5 * nu * ||H||_{F}^2 + 0.5 * mul * ||W||_F^2
        '''
        (ha, hb) = H.shape
        tmp = 0
        for k in range(ha):
            tmp = tmp + (LA.norm(H[k, :], 1)**2 - LA.norm(H[k, :])**2) 
            
        return LA.norm(X - H * W, 'fro') ** 2 + rho * tmp
    
def update_prim_var_by_PALM(X, W_init, H_init, max_iter, tol, rho):
  
    (ha, hb) = H_init.shape
    H_j_pre, W_j_pre = H_init, W_init
    W_j_cur = W_init
 #   I_ha = np.asmatrix(np.eye(ha))
    all_1_mat =np.ones((hb, hb))
    
    for j in range(max_iter):
     
        fz = np.dot(X, W_j_cur.transpose())
        #fm = np.dot(np.dot(W_j_cur.transpose(), W_j_cur), H_j_pre) + rho * (np.dot(all_1_mat, H_j_pre) - H_j_pre)
        fm = np.dot(H_j_pre, np.dot(W_j_cur, W_j_cur.transpose())) + rho * (np.dot(H_j_pre, all_1_mat) - H_j_pre)
       # fm = W_j_cur.transpose() * W_j_cur * H_j_pre + rho * (all_1_mat * H_j_pre - H_j_pre)
   #     print(fz.shape)
   #     print(fm.shape)
   #     print(H_j_pre.shape)
        tmp = fz / fm
   #     print(t.shape)
        H_j_cur = np.multiply(H_j_pre, tmp) + 1e-80
      #  print(H_j_pre.type)
      #  print(tmp.type)
      #  H_j_change = LA.norm(H_j_cur - H_j_pre, 'fro') / LA.norm(H_j_pre, 'fro')
      #  W_j_change = LA.norm(W_j_cur - W_j_pre, 'fro') / LA.norm(W_j_pre, 'fro')
        H_j_pre = np.copy(H_j_cur)
     #   W_j_pre = np.asmatrix(np.copy(W_j_cur))
        
        res_tol = 1
       
    return (W_j_cur, H_j_cur, j + 1, res_tol)
def solve (X, W, H):
    converge = False
    iter_num = 0
    rho = 1e-10
    gamma = 1.6
    while not converge:
        (W, H, inner_iter_num, restol) = update_prim_var_by_PALM(X, W, H, 200, 1e-5, rho)
       
        t = get_nmf_obj(X, W, H)
        t1 = get_fanshu_obj(X, W, H, rho)
    
    
        str = 'The %d-th iteration nmfobjj: ' %iter_num + '%f' %t
   

        str = 'The %d-th iteration fanshuobj: ' %iter_num + '%f' %t1

        iter_num = iter_num + 1
        rho = np.minimum(rho * gamma,  1e20)
        if iter_num > 60: converge = True
       
        else: converge = False
    return (W, H)


# In[20]:


random.seed(15)#测试用10


# In[21]:


def generate_random_id(sign_tot):
    result = []

    while len(result) < sign_tot:
        random_num = random.randint(0, 7)
        if random_num not in result:
            result.append(random_num)
    result.sort()
    return result

 # 定义一个函数，用于提取数据并对齐索引
def extract_and_align(data, indices, total_indices):
    aligned_data = np.full((len(total_indices), 1), np.nan)  # 生成全是 NaN 的数组
    for i, idx in enumerate(total_indices):
        if idx in indices:
            aligned_data[i] = data[idx]  # 如果索引存在于给定的 indices 中，则提取数据
    return aligned_data
# In[22]:
################别动这个!!!!!!!!!!!!!!!!!!!!!!!!!########################
n=5000
###########################
#获取信号的数量

rand_numbers = []
for i in range(n):
    x = random.randint(4, 8)
    #x=4
    rand_numbers.append(x)

# In[23]:

#存储信号位置
id_numbers = []
for i in range(n):
    random_numbers = generate_random_numbers()
    id_numbers.append(random_numbers) 
#存储信号发生功率  
id_p = []
for i in range(n):
    random_numbers = generate_random_p()
    id_p.append(random_numbers) 
#获取此次接收到的信号索引
id_id = []
for i in range(n):
    x = rand_numbers[i]
    random_numbers = generate_random_id(x)
    id_id.append(random_numbers)


# In[25]:


##print(id_numbers[0])
##print(id_p[0])
##print(id_id[0])
##print(id_numbers[0])

#print(data[id_id[0][0]])
#print(data[id_id[1][0]])
#print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# In[26]:


###print(id_id[0:10])


# In[14]:

#无人机高度
id_H = [25, 40, 20, 80, 100, 120, 140]
#信噪比
id_SNR = [10, 9, 7, 8, 9, 10]#10 9  8 7

#########################
#循环次数
test_n =200
#########################
total_res = []
total_id = []
total_p = []
h = np.ones((8,100))
h = np.array(h).reshape((8,100))
#无人机位置
idm1 = 73
idm2 = 28
idm3 = 23
idm4 = 78


# In[15]:

import time
start_time = time.time()
id_h = 2#无人机高度索引
id_snr=0#信噪比索引
#存储符合要求的标准位置
temp_id_numbers=[]
#存储符合要求的标准功率
temp_id_p=[]
v1_d = []
v2_d = []
v3_d = []
v4_d = []

kk = 50 / len2(34,100,id_H[id_h])
##print(kk)

res = 0 
str =  'The %d-th iteration : ' %id_h
##print(str)
count=0
LAST_X = []
lost_index=[]
for i in range(test_n):
    print("begin>>>",i)
    lenlen = rand_numbers[i]#此次获取信号的数量
    print("信号数量>>>",lenlen)
    s_v1 = 0
    s_v2 = 0
    s_v3 = 0
    s_v4 = 0
    data, gnd = mat['fea'].astype('float32'), mat['gnd']
    l1=0
    l2=0
    l3=0
    l4=0
    
    #print("id_id[0]>",id_id[0])
    for k in range(lenlen):
        #print(math.sqrt(len2(idm1, id_numbers[i][k], id_H[id_h])))
        #print(math.sqrt(len2(idm2, id_numbers[i][k], id_H[id_h])))
        #print(math.sqrt(len2(idm3, id_numbers[i][k], id_H[id_h])))
        #print(math.sqrt(len2(idm4, id_numbers[i][k], id_H[id_h])))
        if math.sqrt(len2(idm1, id_numbers[i][k], id_H[id_h]))<78:
            s_v1 = s_v1 + id_p[i][k] / len2(idm1, id_numbers[i][k], id_H[id_h]) * (np.array(data)[id_id[i][k]])
            l1=l1+1
        if math.sqrt(len2(idm2, id_numbers[i][k], id_H[id_h]))<78:
            l2=l2+1
            s_v2 = s_v2 + id_p[i][k] / len2(idm2, id_numbers[i][k], id_H[id_h]) * (np.array(data)[id_id[i][k]])
        if math.sqrt(len2(idm3, id_numbers[i][k], id_H[id_h]))<78:
            l3=l3+1
            s_v3 = s_v3 + id_p[i][k] / len2(idm3, id_numbers[i][k], id_H[id_h]) * (np.array(data)[id_id[i][k]])
        if math.sqrt(len2(idm4, id_numbers[i][k], id_H[id_h]))<78:
            l4=l4+1
            s_v4 = s_v4 + id_p[i][k] / len2(idm4, id_numbers[i][k], id_H[id_h]) * (np.array(data)[id_id[i][k]])
    print(f"numbers_{count}>>>",id_numbers[count])
    print(f"p_{count}>>>",id_p[count])
    #共有lenlen个信号，如果每一个信号都被接收到，则开始下一个循环
    if all(arr == lenlen for arr in [l1, l2, l3, l4]):
        print("信号都存在")
        lost_index.append(i)
        count=count+1
        continue
    if l1==0 or l2==0 or l3==0 or l4==0:
        print("某监测点未接收到任何信号")
        lost_index.append(i)
        count = count + 1
        continue
    s_v1 = np.array(s_v1).reshape((1024,1))
    s_v2 = np.array(s_v2).reshape((1024,1))
    s_v3 = np.array(s_v3).reshape((1024,1))
    s_v4 = np.array(s_v4).reshape((1024,1))
   
    ans =id_id[i]
    q1 = wgn(s_v1, id_SNR[id_snr])
    q2 = wgn(s_v2, id_SNR[id_snr])
    q3 = wgn(s_v3, id_SNR[id_snr])
    q4 = wgn(s_v4, id_SNR[id_snr])
    
    
    ans1 = []
    ans2 = []
    ans3 = []
    ans4 = []
    
    hh1 = []
    hh2 = []
    hh3 = []
    hh4 = []
    #w1是Bs   h是Am
    w1,h1 = nmf(q1, 8, 200)
    for j in range(8):
        if h1[j] > kk:
            ans1.append(j)
    
    w2,h2 = nmf(q2, 8, 200)
    for j in range(8):
        if h2[j] > kk:
            ans2.append(j)
    
    w3,h3 = nmf(q3, 8, 200)
    for j in range(8):
        if h3[j] > kk:
            ans3.append(j)
    
    w4,h4 = nmf(q4, 8, 200)
    for j in range(8):
        if h4[j] > kk:
            ans4.append(j)
    #print(h2)
    #print(ans2)
    total_ans=list(set(ans1)|set(ans2)|set(ans3)|set(ans4))
    #print("total_ans>",total_ans)
    # 保存到 .npz 文件   其中ans[i]表示存在的信号索引下标
    #np.savez(f'../NMFdata&totalindex/nmf_results{count}.npz', NMF_H=[h1, h2, h3, h4], total_index=total_ans)
#------------------------------------保存完整数据--------------------------------
    s_v11 = 0
    s_v22 = 0
    s_v33 = 0
    s_v44 = 0

    for k in range(lenlen):
        s_v11 = s_v11 + id_p[i][k] / len2(idm1, id_numbers[i][k], id_H[id_h]) * (np.array(data)[id_id[i][k]])
        s_v22 = s_v22 + id_p[i][k] / len2(idm2, id_numbers[i][k], id_H[id_h]) * (np.array(data)[id_id[i][k]])
        s_v33 = s_v33 + id_p[i][k] / len2(idm3, id_numbers[i][k], id_H[id_h]) * (np.array(data)[id_id[i][k]])
        s_v44 = s_v44 + id_p[i][k] / len2(idm4, id_numbers[i][k], id_H[id_h]) * (np.array(data)[id_id[i][k]])
    s_v11 = np.array(s_v11).reshape((1024,1))
    s_v22 = np.array(s_v22).reshape((1024,1))
    s_v33 = np.array(s_v33).reshape((1024,1))
    s_v44 = np.array(s_v44).reshape((1024,1))
    
    ans =id_id[i]
    q11 = wgn(s_v11, id_SNR[id_snr])
    q22 = wgn(s_v22, id_SNR[id_snr])
    q33 = wgn(s_v33, id_SNR[id_snr])
    q44 = wgn(s_v44, id_SNR[id_snr])
    
    
    ans11 = []
    ans22 = []
    ans33 = []
    ans44 = []
    
    h11 = []
    h22 = []
    h33 = []
    h44 = []
    #w1是Bs   h是Am
    w11,h11 = nmf(q11, 8, 200)
    for j in range(8):
        if h11[j] > kk:
            ans11.append(j)
    
    w22,h22 = nmf(q22, 8, 200)
    for j in range(8):
        if h22[j] > kk:
            ans22.append(j)
    
    w33,h33 = nmf(q33, 8, 200)
    for j in range(8):
        if h33[j] > kk:
            ans33.append(j)
    
    w44,h44 = nmf(q44, 8, 200)
    for j in range(8):
        if h44[j] > kk:
            ans44.append(j)
    h111=h11
    h222=h22
    h333=h33
    h444=h44

# =============================================================================
#     print('-------------')
#     print(h22)
#     print(ans22)
# =============================================================================
    #获取标准数据
    h11=h11[ans11].flatten()
    h22=h22[ans22].flatten()
    h33=h33[ans33].flatten()
    h44=h44[ans44].flatten()
    
    #如果高精度数据数量判断错误，则开始下一个循环
    if any(len(arr) != lenlen for arr in [h11, h22, h33, h44]):
        print("高精度数据判断错误>>>",i)
        lost_index.append(i)
        print("-----------------------")
        count=count+1
        continue  # 跳过此次循环，开始下一次循环
    temp_id_numbers.append(id_numbers[i])
    temp_id_p.append(id_p[i])
    np.savez(f'../NMFdata&totalindex/nmf_results{count}.npz', NMF_H=[h1, h2, h3, h4], total_index=total_ans,
             location=id_numbers[count],p=id_p[count],NMF_HH=[h111, h222, h333, h444])
    CompleteDataMatrix=np.empty((4,), dtype=object)
    CompleteDataMatrix[0] = h11
    CompleteDataMatrix[1] = h22
    CompleteDataMatrix[2] = h33
    CompleteDataMatrix[3] = h44
    total_ans=list(set(ans1)|set(ans2)|set(ans3)|set(ans4))
    # 分别提取并对齐 h1, h2, h3, h4 中的数据
    h1_aligned = extract_and_align(h1, ans1, total_ans)
    h2_aligned = extract_and_align(h2, ans2, total_ans)
    h3_aligned = extract_and_align(h3, ans3, total_ans)
    h4_aligned = extract_and_align(h4, ans4, total_ans)
    
    #---------------------------------转换成mat文件进行的操作--------------------------
    # 将对齐的数据组合成一个列表 aligned_data中保存的是要补全时的数据，其中缺失的用nan来替代
    aligned_data = [h1_aligned, h2_aligned, h3_aligned, h4_aligned]
    # 初始化二维列表存储结果
    index_list = []
    unfind_index=[]
    # 遍历每个对齐的数据数组，生成存在数据的索引
    #生成存在数据的索引下标
    for data in aligned_data:
        indices = [j  for j, value in enumerate(data) if not np.isnan(value)]
        unindices=[j  for j, value in enumerate(data) if np.isnan(value)]
        index_list.append(indices)
        unfind_index.append(unindices)
    
    
    #展平到1纬
    h1_aligned=h1_aligned.flatten()
    h2_aligned=h2_aligned.flatten()
    h3_aligned=h3_aligned.flatten()
    h4_aligned=h4_aligned.flatten()
    #保存不完整数据
    incompleteData = np.empty((4,), dtype=object)
    incompleteData[0] = h1_aligned*10000
    incompleteData[1] = h2_aligned*10000
    incompleteData[2] = h3_aligned*10000
    incompleteData[3] = h4_aligned*10000
    #print(incompleteData)
    #保存存在索引
    signalIndices = np.empty((4,), dtype=object)
    signalIndices[0] = index_list[0]
    signalIndices[1] = index_list[1]
    signalIndices[2] = index_list[2]
    signalIndices[3] = index_list[3]
    #print(signalIndices)
    #保存缺失索引
    unfindIndex = np.empty((4,), dtype=object)
    unfindIndex[0]=unfind_index[0]
    unfindIndex[1]=unfind_index[1]
    unfindIndex[2]=unfind_index[2]
    unfindIndex[3]=unfind_index[3]
    


    # 将数据保存为 .mat 文件
    sio.savemat(f'../data/IncompleteFusionData{count}.mat', {'incompleteData': incompleteData,'signalIndices':signalIndices,
                                             'CompleteDataMatrix':CompleteDataMatrix,'unfindIndex':unfindIndex})
    
    # 将数据写入txt文件
    with open(f'../data/data_output{count}.txt', 'w') as file:
        # 写入每个模块
        for i, (h_standard, h_missing, missing_idx) in enumerate(zip([h11, h22, h33, h44], [h1_aligned, h2_aligned, h3_aligned, h4_aligned], unfind_index), start=1):
            file.write(f"标准: {h_standard }\n")      # 写入标准数据
            file.write(f"缺失: {h_missing }\n")          # 写入缺失数据
            file.write(f"缺失索引: {missing_idx}\n\n")  # 写入缺失索引
    
    ####################################为了GAN网络保存的数据###########################################
    ####################################保存掩码
    
    # 使用列表生成式创建
    mask = [[0] * lenlen for _ in range(4)]
    # 在没有信号的索引处标记为1
    for i in range(4):
        for index in unfindIndex[i]:
            mask[i][index] = 1
    mask_column = np.concatenate(mask).reshape(4, lenlen)
    
    # 创建 DataFrame
    mask_df = pd.DataFrame(mask_column)
    # 将 DataFrame 保存到 CSV 文件中
    mask_df.to_csv(f'../GAN/M{count}.csv', index=False, header=False)
    ###################################保存H矩阵
    complete_matrix = np.vstack(CompleteDataMatrix)
    
    # 使用 pandas 保存为 CSV 文件
    df = pd.DataFrame(complete_matrix)
    df.to_csv(f'../GAN/H{count}.csv', index=False, header=False)
    count=count+1
print("未存放的数据包括>",len(lost_index))


'''
        for p in range(8):
            hh1.append(h1[p])
            hh2.append(h2[p])
            hh3.append(h3[p])
            hh4.append(h4[p])
            
        v1_d.append(hh1)
        v2_d.append(hh2)
        v3_d.append(hh3)
        v4_d.append(hh4)
       
        TK = []
        TK = np.hstack((h1, h2, h3, h4))
        LAST_X.append(TK)
        
        pj = 0 
        if ans1 == ans: 
            pj = pj + 1
        if ans2 == ans: 
            pj = pj + 1
        if ans3 == ans: 
            pj = pj + 1
        if ans4 == ans: 
            pj = pj + 1
        if pj >= 3: 
            res = res + 1
            
    total_res.append(res)    
    W1 = []
    for i in range(100):
        tid = i + 1
        W1.append(1/len2(idm1,tid,id_H[id_h]))
        
    W2 = []
    for i in range(100):
        tid = i + 1
        W2.append(1/len2(idm2,tid,id_H[id_h]))
        
    W3 = []
    for i in range(100):
        tid = i + 1
        W3.append(1/len2(idm3,tid,id_H[id_h]))

    W4 = []
    for i in range(100):
        tid = i + 1
        W4.append(1/len2(idm4,tid,id_H[id_h]))
        
    W1 = np.array(W1).reshape((100,1))
    W2 = np.array(W2).reshape((100,1))
    W3 = np.array(W3).reshape((100,1))
    W4 = np.array(W4).reshape((100,1))
    
    W = []
    W = np.hstack((W1, W2, W3, W4))
    
    ans_id = []
    ans_p = []
    #print(LAST_X)
    for ii in range(test_n):
       
        X = LAST_X[ii]
        
        (p, q) = solve(X, W, h)
        #print(q)
        ans_idd = []
        ans_idp = []
        
        
        lenlen = rand_numbers[ii]
        for k in range(lenlen):
            idk = np.argmax(q[id_id[ii][k], :])
            idp = q[id_id[ii][k], idk]
            ans_idd.append(idk)
            ans_idp.append(idp)
       # print("lenlen=")
       # print(lenlen)
        ans_id.append(ans_idd)
        ans_p.append(ans_idp)
        print(ii)
      
    tot_id = 0 
    tot_p = 0
    data_id = []
    data_p = []
      
    for i in range(test_n):
        
        lenlen = rand_numbers[i]
        tot1 = 0
        tot2 = 0
        for k in range(lenlen):
            idx1 = ans_id[i][k] + 1
            idy1 = id_numbers[i][k]
            
            idx2 = ans_p[i][k]
            idy2 = id_p[i][k]
            tot1 = tot1 + np.sqrt(obj2(POS(idx1),POS(idy1))) 
            tot2 = tot2 + np.abs(idx2 - idy2) / idy2
  
        tot1 = tot1 / lenlen
        tot2 = tot2 / lenlen
        
        tot_id = tot_id + tot1
        tot_p = tot_p + tot2
        
    total_id.append((tot_id / test_n + 2)/40)
    total_p.append(tot_p / test_n)

end_time = time.time()
execution_time = end_time - start_time
print(f"程序执行时间：{execution_time} 秒")
"""

# In[16]:
"""
print(total_res)
print(total_id)
print(total_p)
"""
'''





