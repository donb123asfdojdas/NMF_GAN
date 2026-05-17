% construct similarity matrix with probabilistic k-nearest neighbors. It is a parameter free, distance consistent similarity.
function W = constructW_PKN(X, k, issymmetric)
% X: each column is a data point
% k: number of neighbors
% issymmetric: set W = (W+W')/2 if issymmetric=1
% W: similarity matrix

if nargin < 3
    issymmetric = 1;
end
if nargin < 2
    k = 5;
end
[dim, n] = size(X);%dim为数据维度,n为样本数量
D = L2_distance_1(X, X);%求矩阵X中各个元素之间的距离
[dumb, idx] = sort(D, 2); %对矩阵每一行进行从小到大排序形成新矩阵，其中idx表示源数据的位置

W = zeros(n);
for i = 1:n
    id = idx(i,2:k+2);%提取第i行第2个到第k+2个元素作为与该元素最近的k个元素
    di = D(i, id);%di 将包含第i个样本与其最近邻的距离。例如，如果 id=[2,3]，而D是距离矩阵，那么 di 将包含D(i,2)和D(i,3)的值。
    %这段代码实际上在计算样本i和其 k个最近邻的相似度。相似度的值会被填入相似度矩阵 W的对应位置 (i,id)中。
    W(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end

if issymmetric == 1
    W = (W+W')/2;
end




% compute squared Euclidean distance
% ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
function d = L2_distance_1(a,b)
% a,b: two matrices. each column is a data
% d:   distance matrix of a and b



if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))]; 
  b = [b; zeros(1,size(b,2))]; 
end

aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;

d = real(d);
d = max(d,0);

% % force 0 on the diagonal? 
% if (df==1)
%   d = d.*(1-eye(size(d)));
% end





