function totalRMSE = calculateTotalRMSE(X, unfindIndex,fileIndex)
    outputFileName = sprintf('data/IncompleteFusionData%d.mat', fileIndex);
    % 加载 CompleteDataMatrix 数据
    load(outputFileName, 'CompleteDataMatrix'); % 只加载 CompleteDataMatrix 变量
    for k = 1:length(unfindIndex)
        unfindIndex{k} = unfindIndex{k} + 1;
    end
    % 初始化总的平方误差和
    totalSquaredError = 0;
    totalErrorCount = 0; % 总的误差数据点数
    % 遍历每个预测值和标准值
    numPredictions = length(X); % 预测值的数量
    for i = 1:numPredictions
        % 提取预测值和标准值
        prediction = X{i}; % 获取预测值向量
        standard = CompleteDataMatrix{i}; % 使用 {} 获取标准值向量

        % 检查向量大小是否一致
        if length(prediction) == length(standard)
            % 获取当前视角中需要计算误差的索引
            incorrectIndices = unfindIndex{i};

            % 如果该视角中没有指定的错误数据，跳过
            if isempty(incorrectIndices)
                continue;
            end
            
            % 累加指定位置的平方误差
            squaredErrors = (prediction(incorrectIndices) - standard(incorrectIndices)).^2;
            totalSquaredError = totalSquaredError + sum(squaredErrors); % 累加平方误差
            totalErrorCount = totalErrorCount + length(incorrectIndices); % 累加误差数据点数
        else
            warning('第 %d 个预测值与标准值长度不匹配，跳过。', i);
        end
    end

    % 计算总的均方根误差
    if totalErrorCount > 0
        totalRMSE = sqrt(totalSquaredError / totalErrorCount);
    else
        warning('未找到需要计算误差的数据。');
        totalRMSE = NaN;
    end
end
