  % 参数设置
numViews =4;           % 视角数量（无人机数量）
numCols = 6;            % 信号源数量
gridSize = 10;          % 一行网格数量
planeSize = 120;        % 平面大小（假设为400x400）
gridWidth = planeSize / gridSize; % 每个网格的宽度
limit = 100;             % 距离限制
high = 60;              % 无人机高度
unfindIndex = cell(numViews, 1);

% 初始化信号源和无人机的位置坐标矩阵
signalPositions = zeros(2, numCols); % 信号源位置矩阵
uavPositions = zeros(2, numViews);   % 无人机位置矩阵
occupiedGrids = false(gridSize, gridSize); % 跟踪占用的网格，避免重叠

% 使用所有网格中心点作为kmeans的输入数据
gridCenters = [];
for row = 1:gridSize
    for col = 1:gridSize
        x = (col - 0.5) * gridWidth;
        y = (row - 0.5) * gridWidth;
        gridCenters = [gridCenters; x, y];
    end
end

% 使用kmeans聚类，得到numViews个无人机位置
[~, uavPositionsClustered] = kmeans(gridCenters, numViews);
uavPositions = uavPositionsClustered';

% 将无人机位置调整到对应的网格中心坐标 
for i = 1:numViews
    row = round(uavPositions(2, i) / gridWidth + 0.5);
    col = round(uavPositions(1, i) / gridWidth + 0.5);
    
    % 将无人机位置更新为网格中心
    uavPositions(1, i) = (col - 0.5) * gridWidth;
    uavPositions(2, i) = (row - 0.5) * gridWidth;
    
    % 标记该网格为已占用
    occupiedGrids(row, col) = true;
end

% 随机生成信号源的位置，并放置到对应网格中心，避免与无人机重叠
for i = 1:numCols
    while true
        row = randi(gridSize); % 随机选择行
        col = randi(gridSize); % 随机选择列

        % 若该网格未占用，则放置信号源到此网格中心
        if ~occupiedGrids(row, col)
            signalPositions(:, i) = [(col - 0.5) * gridWidth; (row - 0.5) * gridWidth];
            occupiedGrids(row, col) = true; % 标记该网格已占用
            break;
        end
    end
end

%% 计算信号与无人机之间的距离并保存数据
incompleteData = cell(numViews, 1);
signalIndices = cell(numViews, 1);
CompleteDataMatrix = cell(numViews, 1);

% 打开文件准备写入
fid = fopen('data/UAV_signals_with_incomplete_data.txt', 'w');

for i = 1:numViews
    % 初始化完整距离矩阵
    completeSignalMatrix = NaN(1, numCols);
    % 计算每个信号源到该无人机的三维距离
    distances = sqrt(sum((uavPositions(:, i) - signalPositions).^2, 1) + high^2); 
    completeSignalMatrix(1, :) = distances; % 存储完整距离矩阵
    signalMatrix = completeSignalMatrix; % 复制完整距离矩阵
    
    % 将超过限制的距离置为NaN，表示信号丢失
    signalMatrix(signalMatrix > limit) = NaN;
    
    % 记录有效接收信号的索引
    receivedSignals = find(~isnan(signalMatrix));
    signalIndices{i} = receivedSignals;
    missingSignals = find(isnan(signalMatrix)); % 记录缺失信号的索引
    unfindIndex{i} = missingSignals; % 将缺失信号的索引保存到 unfindIndex
    incompleteData{i} = signalMatrix;
    CompleteDataMatrix{i} = completeSignalMatrix;

    % 写入文件
    fprintf(fid, 'View %d\n', i);
    fprintf(fid, 'Received Data:\n');
    fprintf(fid, '%g\t', signalMatrix(1, :));
    fprintf(fid, '\n');
    
    fprintf(fid, 'Existing Data Index:\n');
    fprintf(fid, '%d\t', signalIndices{i});
    fprintf(fid, '\n');
    
    fprintf(fid, 'Complete Data Matrix:\n');
    fprintf(fid, '%g\t', completeSignalMatrix(1, :));
    fprintf(fid, '\n\n');
end

% 关闭文件
fclose(fid);

% 保存数据到 .mat 文件
save('data/UAV_signals_with_incomplete_data.mat', 'incompleteData', 'signalIndices', 'CompleteDataMatrix');

%% 绘制网格、信号源和无人机位置
figure;
hold on;

% 绘制网格线
for x = gridWidth:gridWidth:planeSize
    plot([x, x], [0, planeSize], 'k--'); % 垂直网格线
end
for y = gridWidth:gridWidth:planeSize
    plot([0, planeSize], [y, y], 'k--'); % 水平网格线
end

% 绘制信号位置（红色点）
scatter(signalPositions(1, :), signalPositions(2, :), 100, 'r', 'filled', 'DisplayName', 'Signal Positions');

% 绘制无人机位置（蓝色点）
scatter(uavPositions(1, :), uavPositions(2, :), 100, 'b', 'filled', 'DisplayName', 'UAV Positions');

% 添加图例和标签
legend('show');
xlabel('X Position');
ylabel('Y Position');
title('Signal and UAV Positions in 2D Plane with Grid');
axis equal;
grid on;

% 保存图像
saveas(gcf, 'data/UAV_signals_with_incomplete_data.png');

% 提示完成信息
disp("UAV_signals_with_incomplete_data.txt complete");
disp("UAV_signals_with_incomplete_data.mat complete");
disp("UAV_positions_plot.png complete");

%% 计算并保存所有可能网格中心之间的距离
% 初始化网格中心点坐标
numGrids = gridSize * gridSize;
gridCenters = zeros(2, numGrids);
index = 1;
for row = 1:gridSize
    for col = 1:gridSize
        gridCenters(:, index) = [(col - 0.5) * gridWidth; (row - 0.5) * gridWidth];
        index = index + 1;
    end
end

% 计算所有网格中心点之间的距离并去重、排序
distances = [];
for i = 1:numGrids
    for j = i+1:numGrids % 只计算一次每对距离，避免重复
        dist = sqrt((gridCenters(1, i) - gridCenters(1, j))^2 + ...
                    (gridCenters(2, i) - gridCenters(2, j))^2+high^2);
        distances = [distances, dist]; % 存储距离
    end
end

% 去重并排序
uniqueDistances = unique(distances); % 去重
sortedDistances = sort(uniqueDistances); % 排序

disp(unfindIndex);

% 保存到 .mat 文件
save('data/mtach_distance.mat', 'sortedDistances', 'unfindIndex');
disp("mtach_distance.mat complete");
