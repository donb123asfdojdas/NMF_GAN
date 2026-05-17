import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy import linalg as LA
import numpy as np
import random
import pandas as pd
import math
import scipy.io as sio
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
def wgn(x, SNR):
    noise_m = np.random.randn(1024)
    noise_power = np.sum(x) / (10**(SNR/10)) * (noise_m**2) / np.linalg.norm(noise_m)**2    
    noise_power = np.array(noise_power).reshape((1024, 1))
    signal_add_noise = x + noise_power
    return signal_add_noise

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
        random_num = random.randint(50, 100)
        result.append(random_num)
    
    return result

random.seed(35)

# 定义函数 len2
def obj2(id1, id2):
    # 计算两个网格的坐标
    a = POS(id1)
    b = POS(id2)
    # 计算两点之间的欧几里得距离
    L = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    return np.sqrt(L)  # 返回距离
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

###########################
#获取信号的数量

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

   # W = np.random.rand(xRow, r)
  #  W = justNorm(W)
  ################为了保证对比数据和第一次生成数据相同########################

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

#无人机高度
id_H = [25, 40,30, 80, 100, 120, 140]
#信噪比
id_SNR = [10, 1, 2, 3, 4, 5, 6, 7]

#########################
#循环次数
test_n = 1
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
id_h = 2#无人机高度索引
# 设置目录路径
directory = '../NMFdata&totalindex/'

# 获取所有以 nmf_results 开头的文件
files = [f for f in os.listdir(directory) if f.startswith('nmf_results') and f.endswith('.npz')]
NMF_H=[]
# 存储每组的平均差值
p_mean_differences = []
location_mean_differences=[]
# 遍历文件并加载数据
index=0
for file in files:
    print(file)
    # 提取 count 值（假设文件名格式固定为 nmf_results{count}.npz）
    count = file.replace('nmf_results', '').replace('.npz', '')

    # 构建对应的 CSV 文件名
    csv_file = f"output/output{count}.csv"
    csv_path = os.path.join(directory, csv_file)

    # 检查 CSV 文件是否存在
    if not os.path.exists(csv_path):
        print(f"对应的 CSV 文件 {csv_file} 不存在，跳过此文件。")
        continue

    # 加载 .npz 文件数据
    file_path = os.path.join(directory, file)
    data = np.load(file_path)

    #原始的基于NMF分离出来的h
    NMF_H = data['NMF_H']
    NMF_HH=data['NMF_HH']
    #表示存在的信号的索引0-7，用于替换NMF_H中对应列的数据
    total_index = data['total_index']
    #此次随机生成的8个位置    
    location=data['location']
    #print("location>",location)
    #此次随机生成的8个功率
    id_p=data['p']
    #print("p>",id_p)
    NMF_H=NMF_H.reshape(4,8)
    NMF_HH=NMF_HH.reshape(4,8)
    #print("NMF_H>",NMF_H)
    print("total_index>",total_index)
    # 加载 CSV 数据

    csv_data = pd.read_csv(csv_path, header=None).values  # 假设无表头
    # 验证长度一致性
    if len(total_index) != csv_data.shape[1]:
        raise ValueError(f"{file} 和 {csv_file} 的数据长度不一致！")
    # 更新 NMF_H 数据
    for i, idx in enumerate(total_index):
        NMF_H[:, idx] = csv_data[:, i]  # 替换 NMF_H 的对应列数据
    #print("处理完后》",NMF_H)
    #print("标准数据》",NMF_HH.reshape(4,8))
    TK = []
    ##########################注意，TK是开关，NMF_H是补全与标准的对比，NMF_HH则是完整的与标准对比########
    TK =NMF_H
    LAST_X=[]
    LAST_X.append(TK)
       
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
    for ii in range(test_n):
        X = LAST_X[ii]
        #q中包含位置和功率
        (p, q) = solve(X.T, W, h)
        ans_idd = []
        ans_idp = []
        #print("total_index>",total_index)
        lenlen =len(total_index) #rand_numbers[ii]
        #存在几个信号取几次
        id_id=[total_index]
        for k in range(lenlen):
            #id_id为此次接收到的信号索引
            idk = np.argmax(q[id_id[ii][k], :])
            idp = q[id_id[ii][k], idk]
            idk=idk+1
            ans_idd.append(idk)
            ans_idp.append(idp)
        # print("lenlen=")
        # print(lenlen)
        
        ans_id.append(ans_idd)
        ans_p.append(ans_idp)
    #print("ans_idd>",ans_idd)
    #print("ans_idp>",ans_idp)
    #print(total_index)
    #处理功率误差
    id_p=np.array(id_p)
    ans_p=np.array(ans_p)#预测功率
    id_p=id_p[:lenlen]
    difference=np.mean(np.abs(id_p-ans_p)/id_p)
#    print("标准功率》",id_p)
  #  print("预测功率》",ans_p)
    p_mean_differences.append(difference)
    #处理距离误差
    location=np.array(location)
    ans_id=np.array(ans_id).flatten() 
    location=location[:lenlen]
    errors = np.mean([obj2(loc, ans) for loc, ans in zip(location, ans_id)])
    location_mean_differences.append(errors)
    print(f"{index}组功率误差>",difference)
    print(f"{index}组距离误差>",errors)
    print("标准数据>",location)
    print("补全获得>",ans_id)
    print("标准功率>", id_p)
    print("预测功率>", ans_p)
    index=index+1
p_total_differences=np.mean(p_mean_differences)
location_total_differences=(np.mean(location_mean_differences)+4)/80
print("距离平均误差为：",location_total_differences)
print("功率平均误差为：",p_total_differences)

'''
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

'''

"""
SNR=14
Stand
距离平均误差为： 0.08157138756706403
功率平均误差为： 0.03520710811818741
NNM(only)
距离平均误差为： 0.19958515529110046
功率平均误差为： 0.07564867519668414
FNCC
距离平均误差为： 0.13483233559007018
功率平均误差为： 0.05724761203571641
DMF
距离平均误差为： 0.17435597152178206
功率平均误差为： 0.06413301074273886
AEMCNE
距离平均误差为： 0.15519440395310266
功率平均误差为： 0.056356289364225993
HOC 
距离平均误差为： 0.3030635098623762
功率平均误差为： 0.12402786641829819
SVD
距离平均误差为： 0.3515656596626196
功率平均误差为： 0.09390635003350549
NNM-GAN
距离平均误差为： 0.13272949595486416
功率平均误差为： 0.04985256108123841

SNR=12
Stand
距离平均误差为： 0.08139107102367013
功率平均误差为： 0.03828660629123745
NNM(only)
距离平均误差为： 0.19762733084433806
功率平均误差为： 0.07857380827349986
FCNN
距离平均误差为： 0.13418004756109386
功率平均误差为： 0.05756836956711093
DMF
距离平均误差为： 0.17375135517406806
功率平均误差为： 0.06455282138208032
AEMCNE
距离平均误差为： 0.15260410857429058
功率平均误差为： 0.05766356400937123
HOC 
距离平均误差为： 0.30426416346416168
功率平均误差为： 0.12858761316843161
SVD
距离平均误差为： 0.3526461316461354
功率平均误差为： 0.09390635003350549
NNM-GAN
距离平均误差为： 0.13037789132913321
功率平均误差为： 0.05242819700275575

SNR=10
Stand
距离平均误差为： 0.08093706987799305
功率平均误差为： 0.04743547337774491
NNM(only)
距离平均误差为： 0.19999176089319223
功率平均误差为： 0.08625581734789482
FCNN
距离平均误差为： 0.13362048333374296
功率平均误差为： 0.06279292013506119
DMF
距离平均误差为： 0.17395195216510578
功率平均误差为： 0.07083821504569691
AEMCNE
距离平均误差为： 0.15521851365536515
功率平均误差为： 0.06327025141598168
HOC
距离平均误差为： 0.30573442715655197
功率平均误差为： 0.1332769811302606
SVD
距离平均误差为： 0.3535496047383734
功率平均误差为： 0.10549542305802234
NNM-GAN
距离平均误差为： 0.12976259865249987
功率平均误差为： 0.05798984340166884

SNR=8
Stand
距离平均误差为： 0.08967337555907376
功率平均误差为： 0.06321887277807535
NNM(only)
距离平均误差为： 0.2029359053837995
功率平均误差为： 0.09802604636771906
FCNN
距离平均误差为： 0.13348070458288272
功率平均误差为： 0.07294143225313492
DMF
距离平均误差为： 0.17697573314112375
功率平均误差为： 0.08070644224606847
AEMC-NE
距离平均误差为： 0.1545450144204536
功率平均误差为： 0.07442190740385955
HOC
距离平均误差为： 0.3066318930573834
功率平均误差为： 0.14441546678994546
SVD
距离平均误差为： 0.35702644330929034
功率平均误差为： 0.1154716987084186
NNM-GAN
距离平均误差为： 0.13117111391806904
功率平均误差为： 0.06882772406179899

SNR=6
Stand
距离平均误差为： 0.10274747491876038
功率平均误差为： 0.08939836544480066
NNM(only)
距离平均误差为： 0.21450521390184168
功率平均误差为： 0.11807202917501569
FCNN
距离平均误差为： 0.14510422731333805
功率平均误差为： 0.09286484207925083
DMF
距离平均误差为： 0.18491857536617445
功率平均误差为： 0.09274289672074973
AEMCNE
距离平均误差为： 0.15201154302127023
功率平均误差为： 0.08969129680071267
HOC
距离平均误差为： 0.30865189421173532
功率平均误差为： 0.15641564464216646
SVD
距离平均误差为： 0.36416574931164921
功率平均误差为： 0.12349656216483168
NNM-GAN
距离平均误差为： 0.13531851601786524
功率平均误差为： 0.08824968071288128

SNR=4
Stand
距离平均误差为： 0.10843075471139518
功率平均误差为： 0.10230964552707539
NNM(only)
距离平均误差为： 0.2164455717195925
功率平均误差为： 0.13275639579757087
FCNN
距离平均误差为： 0.1428936738898683
功率平均误差为： 0.1132007398909004
DMF
距离平均误差为： 0.18491610429449012
功率平均误差为： 0.1251913629122018
AEMCNE
距离平均误差为： 0.1598083013942091
功率平均误差为： 0.1168015144385702
HOC
距离平均误差为： 0.31065189421173532
功率平均误差为： 0.16252578223684352
SVD
距离平均误差为： 0.36984643468716616
功率平均误差为： 0.13494634964616846
NNM-GAN
距离平均误差为： 0.13938144156601937
功率平均误差为： 0.10961061198530483

"""


'''
SNR=10  
HIGHT=30
Stand
距离平均误差为： 0.1439787162529224
功率平均误差为： 0.0802304705339297
NNM(only)
距离平均误差为： 0.14211388993110036
功率平均误差为： 0.07557318437078869
DMF
距离平均误差为： 0.1421113824557238
功率平均误差为： 0.07434355559924472
AEMCNE
距离平均误差为： 0.1432028929638057
功率平均误差为： 0.07733533361907169
HOC
距离平均误差为： 0.14050003786581153
功率平均误差为： 0.06515698959761267
SVD
距离平均误差为： 0.2771587575477076
功率平均误差为： 0.12990435567218786
NNM-GAN
距离平均误差为： 0.1423668667560131
功率平均误差为： 0.07597588747268397
HIGHT =40
Stand
距离平均误差为： 0.07652541734279072
功率平均误差为： 0.033050479672969535
NNM(only)
距离平均误差为： 0.07837101501441131
功率平均误差为： 0.03683303916891106
DMF
距离平均误差为： 0.07735622802942711
功率平均误差为： 0.03611543909653285
AEMCNE
距离平均误差为： 0.07626094374977879
功率平均误差为： 0.033769864252506845
HOC
距离平均误差为： 0.0911380618928518
功率平均误差为： 0.07134506883875452
SVD
距离平均误差为： 0.2634425708489319
功率平均误差为： 0.08530675212903496
NNM-GAN
距离平均误差为： 0.07683553836611694
功率平均误差为： 0.03665310821670699
HIGHT=50
Stand
距离平均误差为： 0.06732841161263022
功率平均误差为： 0.03742893130508153
NNM(only)
距离平均误差为： 0.08231220503527278
功率平均误差为： 0.047570444410980066
DMF
距离平均误差为： 0.08397685290445231
功率平均误差为： 0.055100233182797664
AEMCNE
距离平均误差为： 0.07962122437433902
功率平均误差为： 0.048294443362221155
HOC
距离平均误差为： 0.11065380033899289
功率平均误差为： 0.09783552719609497
SVD
距离平均误差为： 0.17361229630166817
功率平均误差为： 0.08941637749243639
NNM-GAN
距离平均误差为： 0.07779689891282379
功率平均误差为： 0.044297351459858296
HIGHT=60
Stand
距离平均误差为： 0.08093706987799305
功率平均误差为： 0.04743547337774491
NNM(only)
距离平均误差为： 0.19999176089319223
功率平均误差为： 0.08625581734789482
DMF
距离平均误差为： 0.17395195216510578
功率平均误差为： 0.07083821504569691
AEMCNE
距离平均误差为： 0.15521851365536515
功率平均误差为： 0.06327025141598168
HOC
距离平均误差为： 0.30573442715655197
功率平均误差为： 0.1332769811302606
SVD
距离平均误差为： 0.3535496047383734
功率平均误差为： 0.10549542305802234
NNM-GAN
距离平均误差为： 0.12976259865249987
功率平均误差为： 0.05798984340166884
HIGHT=70
Stand
距离平均误差为： 0.08997411264381658
功率平均误差为： 0.04953403597677858
NNM(only)
距离平均误差为： 0.31825341443595734
功率平均误差为： 0.08423124360062435
DMF
距离平均误差为： 0.301494187575338
功率平均误差为： 0.13263180841266854
AEMCNE
距离平均误差为： 0.2842294405068217
功率平均误差为： 0.115604264269737
HOC
距离平均误差为： 0.3820031296484718
功率平均误差为： 0.0909669398813616
SVD
距离平均误差为： 0.39820074690501117
功率平均误差为： 0.09001538281063905
NNM-GAN
距离平均误差为： 0.20133624828992445
功率平均误差为： 0.05289619578395199
'''

