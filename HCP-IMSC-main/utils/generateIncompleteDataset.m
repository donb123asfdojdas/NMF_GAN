function [data, index] = generateIncompleteDataset(numViews, numRows, numCols)
    % 初始化 data 和 index
    data = cell(numViews, 1);
    index = cell(numViews, 1);
    
    % 打开一个文件用于写入所有视图的数据和索引
    combinedFileName = 'combined_data_and_index.txt';
    fileID = fopen(combinedFileName, 'w');  % 以写模式打开文件
    
    for v = 1:numViews
        % 1. 生成完整的数据矩阵
        data{v} = rand(numRows, numCols);
        
        % 2. 确定缺失数据的索引
        % 假设每个样本至少有 30% 的数据是缺失的
        missingRatio = 0.3;
        numMissing = round(missingRatio * numCols);  % 每列30%缺失
        
        % 检查 numMissing 不超过总列数
        if numMissing > numCols
            numMissing = numCols;
        end
        
        % 随机选择缺失数据的列索引
        colIndicesMissing = randperm(numCols, numMissing);  % 随机选择缺失列的索引
        
        % 将这些列的所有行设为 NaN（即整列缺失）
        for i = 1:length(colIndicesMissing)
            data{v}(:, colIndicesMissing(i)) = NaN;
        end
        
        % 3. 存储存在数据的列索引
        colIndicesExisting = setdiff(1:numCols, colIndicesMissing);  % 获取存在数据的列索引
        index{v} = colIndicesExisting;  % 保存存在数据的列索引
        
        % 4. 写入文件
        fprintf(fileID, 'View %d\n', v);  % 写入视图编号
        
        % 写入数据矩阵
        fprintf(fileID, 'Data:\n');
        for i = 1:numRows
            fprintf(fileID, '%f\t', data{v}(i, :));
            fprintf(fileID, '\n');
        end
        
        % 写入存在数据的列索引
        fprintf(fileID, 'Existing Data Index:\n');
        fprintf(fileID, '%d\t', index{v});
        fprintf(fileID, '\n\n');
    end
    
    % 关闭文件
    fclose(fileID);
end
