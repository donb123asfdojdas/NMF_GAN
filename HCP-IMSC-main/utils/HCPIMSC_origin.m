function [R, history, X, omega, G, Z] = HCPIMSC(Xo, Po, Xu, Pu, lambda1, lambda2, c)
% Inputs:
%   Xo - observed parts, a cell array, num_view*1, each array is d_v*n_v
%   Po - projection matrices, a cell array, num_view*1, each array is n_v*n
%   Xu - missing parts, a cell array, num_view*1, each array is d_v*(n-n_v)
%   Pu - projection matrices, a cell array, num_view*1, each array is (n-n_v)*n
%   lambda1,lambda2 - hyperparameters for the algorithm
%   c - number of clusters
% Outputs:
%   R - unified affinity matrix, a n*n array
%   history - reconstruction errors
%   X: reconstructed samples
%   omega - weights of views
%   G - refined view-specific affinity matrices, a cell array, num_view*1, each array is n*n
%   Z - view-specific affinity matrices, a cell array, num_view*1, each array is n*n
% Zhenglai Li, Chang Tang, Xinwang Liu, Xiao Zheng, Wei Zhang, and En Zhu: High-order Correlation Preserved Incomplete Multi-view Subspace Clustering. IEEE Transactions on Image Processing (TIP)
%parameter initial
num_view = length(Xo);%共有多少视角
N = size(Po{1}, 2);%获取的信号个数
%matrix initial
Z = cell(num_view, 1);
G = cell(num_view, 1);
X = cell(num_view, 1);
Xc = cell(num_view, 1);
U = cell(num_view, 1);
V = cell(num_view, 1);
Usq = cell(num_view, 1);
R = zeros(N, N);
for v = 1 : num_view
    Z{v} = zeros(N, N);
    G{v} = zeros(N, N);
    X{v} = Xo{v} * Po{v};
    Xc{v} = Xo{v} * Po{v};
end
G_tensor = zeros(N, N, v);
omega = ones(v, 1) ./ v;%初始化权重向量，使得每一个权重的值都为1/v
for iter = 1 : 20
    %     fprintf('----processing iter %d--------\n', iter);
    % update Z
    tempZ = zeros(N, N);%全零的N×N矩阵tempZ，用于累加每个视图的相似度矩阵
    for v = 1 : num_view
        tmp = X{v}' * X{v};%计算内积从而获得X{v}内部数据间的相似度
        Z{v} = ((omega(v) + lambda2) * eye(N, N) + tmp)\(tmp + omega(v) * R + lambda2 * G{v});
        Z{v} = Z{v} - diag(diag(Z{v}));%从 Z{v} 中去掉对角线上的元素，以确保相似度矩阵不包含样本与自身的相似度。
        %======================没有归一化怎么办==================
        Z{v} = max(0.5 * (Z{v} + Z{v}'), 0 );%转置相加，保证对称性，保证值小于等于1，最小值为0
        tempZ = tempZ + omega(v) * Z{v};%将当前视图的加权相似度矩阵 Z{v} 加入到 tempZ 中
    end
    tempZ = tempZ ./ sum(omega);
    R = tempZ - diag(diag(tempZ));%去除对角线上元素，不包含样本与自身的相似度
    R = max(0.5 * (R + R'), 0);
    % update omega
    for v = 1 : num_view
        omega(v) = 0.5 / (norm(Z{v} - R, 'fro') + eps);%使用Frobenius范数，评判Z{v}与R之间的相似度，相似度越高，对应的omega越大
    end
    % update L
    if iter == 1%如果是第一次迭代
        Weight = constructW_PKN(R, 15);%获取到样本中不同视角之间最相近的k个样本的相似度
        Diag_tmp = diag(sum(Weight));%获得对角矩阵，其中每行对应元素为该样本与其最相近k个样本相似度之和
        L = Diag_tmp - Weight;%构建拉普拉斯矩阵，提供样本之间的关系
    else
        param.num_view = 15; 
        HG = gsp_nn_hypergraph(R', param);%调用gspbox来构建超图
        L = HG.L;%基于超图获取超拉普拉斯矩阵
    end
    % update Xu
    M = cell(num_view, 1);
    for v = 1 : num_view
        M{v} = (Z{v} - eye(N)) * (Z{v} - eye(N))' + lambda1 * L;%M{v}存储了样本之间的相似度信息和结构信息的结合
        Xu{v} = ( - Xo{v} * Po{v} * M{v} * Pu{v}') / (Pu{v} * M{v} * Pu{v}' );
        [Xu{v}] = NormalizeData(Xu{v});
        % update X
        X{v} = Xc{v} + Xu{v} * Pu{v};
    end
    % update G
    Z_tensor = cat(3, Z{ : , : });%将Z第三个维度结合在一起，形成三维张量
    hatZ = fft(Z_tensor, [], 3);%对第三维度进行短时傅里叶变换，提取频率特征，便于后续SVD分解
    if iter == 1
        for v = 1 : num_view
            [Unum_view, Sigmanum_view, Vnum_view] = svds(hatZ( : , : , v), c);%对第v个切片进行奇异值分解，保留前c个奇异值对应的向量
            %这里将左奇异向量 Unum_view 和奇异值 Sigmanum_view 结合起来，形成一个矩阵 UvU_vUv​，它表示了当前视图的主要成分。
            U{v} = Unum_view * Sigmanum_view;
            V{v} = Vnum_view';
            %构造出视图特定的亲和矩阵 Gv​，并将其存储在 Gtensor中的第v个切片。
            G_tensor( : , : , v) = U{v} * V{v};
        end
    else
        for v = 1 : num_view
            U{v} = hatZ( : , : , v) * V{v}' * pinv(V{v} * V{v}');
            Usq{v} = U{v}' * U{v};
            V{v} = pinv(Usq{v}) * U{v}' * hatZ( : , : , v);
            G_tensor( : , : , v) = U{v} * V{v};
        end
    end
    G_tensor = ifft(G_tensor, [], 3);
    for v = 1 : num_view
        G{v} = G_tensor( : , : , v);
    end
    %record the iteration information
    history.term1(iter) = 0;
    % coverge condition
    for v = 1 : num_view
        history.term1(iter) = history.term1(iter) + norm(X{v} - X{v} * Z{v}, 'fro') ^ 2 ;
    end
    obj(iter) = history.term1(iter);
    if iter > 2 && abs((obj(iter) - obj(iter - 1)) / obj(iter - 1)) < 1e-4
        break;
    end
end
end