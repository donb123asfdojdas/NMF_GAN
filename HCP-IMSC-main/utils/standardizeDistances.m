function X_standardized = standardizeDistances(X, unfindIndex, sortedDistances)
    % 标准化无人机到节点的距离数据
    % 输入：
    %   X - 包含各视角预测距离的单元数组
    %   unfindIndex - 不准确数据的索引，每个视角中需要修正的索引数组
    %   sortedDistances - 从小到大排序的标准距离值
    % 输出：
    %   X_standardized - 修正后的距离数据单元数组
    
    % 初始化返回值
    X_standardized = X;

    % 遍历 unfindIndex 中的每个视角
    for viewIndex = 1:length(unfindIndex)
        % 获取当前视角中需要修正的数据索引
        incorrectIndices = unfindIndex{viewIndex};
        
        % 如果当前视角中没有需要修正的数据，跳过
        if isempty(incorrectIndices)
            continue;
        end
        
        % 遍历当前视角中的每个需要修正的索引
        for j = 1:length(incorrectIndices)
            dataIndex = incorrectIndices(j); % 当前需要修正的数据的索引
            % 获取需要修正的数据
            incorrectValue = X{viewIndex}(dataIndex);
            % 在 sortedDistances 中找到最接近 incorrectValue 的值
            [~, closestIndex] = min(abs(sortedDistances - incorrectValue));
            closestValue = sortedDistances(closestIndex);
            
            % 替换为最接近的标准值
            X_standardized{viewIndex}(dataIndex) = closestValue;
        end
    end
end
