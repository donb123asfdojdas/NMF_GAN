function [X1, O1, X2, O2] = DataPreparing(data, index)
% X{k} = X1{k}*O1{k} + X2{k}*O2{k}
% [D, N] = size(X{k}), [D, N_1] = size(X1{k}), [D, N_i] = size(X2{k}),
% [N_1, N] = size(O1{k}), [N_i, N] = size(O2{k})

K = length(data); %numOfView
N = size(data{1}, 2); %numOfSample
X1 = cell(K,1); %the complete parts
X2 = cell(K,1); %the missing parts
O1 = cell(K, 1);
O2 = cell(K, 1);
Xc = cell(K,1);
% 将index中的每个单元格中的值全部加1(PYTHON索引从0开始，而matlab是从1开始的)
for k = 1:length(index)
    index{k} = index{k} + 1;
end
for k = 1:K
    data{k} = data{k};
    W1 = ones(N, 1);%生成全为1的矩阵
    W1(index{k}, 1) = 0;%将观测到的位置的设置为0

    ind_1 = W1 == 1;%基于W1生成逻辑数组ind_1，其中未记录的位置为true
    W2 = eye(N);%生成NxN的单位矩阵
    W2(ind_1, :) = [];%删除ind_1中对应为true的行，生成的矩阵为MxN
    O1{k} = W2;%获取观测到矩阵的映射矩阵O1

    W3 = zeros(N, 1);
    W3(index{k}, 1) = 1;
    
    
    ind_2 = W3 == 1;
    W4 = eye(N);
    W4(ind_2, :) = [];
    O2{k} = W4;
    
    data{k} = double(data{k});
    data{k}(isnan(data{k})) = 0;%将NAN的数据转为0
    Xc{k} = data{k} * O1{k}';%获取到观测的部分
    %========================没有归一化怎么办=======================
    %%%%%%%%%%[X1{k}] = NormalizeData(Xc{k});%归一化处理
    X1{k}=Xc{k};
    fillV = repmat(mean(X1{k}, 2), 1, size(O2{k}, 1));
   
    %mean表示将X1基于列求均值，最后得到每一个特征的均值
    %size表示获得O2的行数，即缺失样本的数量
    %repmat表示将mean构建的矩阵重复size次，
    % 生成D x M的矩阵，其中 M 是缺失样本的数量，每一列都是相应特征的均值。
    %%%%%%%%%%%%%%[X2{k}] = NormalizeData(fillV);
    X2{k}=fillV;




    
    % 输出每个视图的维度信息
    %fprintf('View %d\n', k);
    %fprintf('Size of data{k}: %d  %d\n', size(data{k}));
    % 输出每个视图的维度信息
    %fprintf('Size of X1{k}: %d  %d\n', size(X1{k}));
    %fprintf('Size of X2{k}: %d  %d\n', size(X2{k}));
    %fprintf('Size of O1{k}: %d  %d\n', size(O1{k}));
    %fprintf('Size of O2{k}: %d  %d\n', size(O2{k}));
   
end
end