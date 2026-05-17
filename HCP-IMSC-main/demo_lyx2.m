addpath(genpath('gspbox-0.7.0/'));
addpath('ClusteringMeasure', 'LRR', 'utils');

% Specify the number of files to process
numFiles =200;  % Replace 'n' with the actual number of files
% Loop over each file
for fileIndex = 0:numFiles-1
    % Load the current data file
    dataFileName = sprintf('data/IncompleteFusionData%d.mat', fileIndex);
     try
        load(dataFileName);
        % 如果成功加载，可以在这里继续进行数据处理
    catch
        % 如果文件不存在，跳过当前循环，继续下一个
        fprintf('File %s not found. Skipping...\n', dataFileName);
        continue;
    end
    
    numViews = length(incompleteData);
    num=length(incompleteData{1});
    disp(num)
    incompleteDataCell = cell(numViews, 1);   % Initialize cell for data storage
    incompleteIndicesCell = cell(numViews, 1); % Initialize cell for indices storage
    
    for i = 1:numViews
        incompleteDataCell{i} = incompleteData{i}; % Store each view's complete data
        incompleteIndicesCell{i} = signalIndices{i}; % Store each view's index
    end
    
    [X1, O1, X2, O2] = DataPreparing(incompleteDataCell, incompleteIndicesCell);
    TempLambda1 = 0.001;
    TempLambda2 = 0.001;
    c = 18;
    
    % Set up output filenames
    %outputFileName = sprintf('data/completed_matrices_combined%d.txt', fileIndex);
    csvOutputFileName = sprintf('GAN/L%d.csv', fileIndex);
    
    perfect_pross = Inf;
    pmin_error = 0;
    
    % Matrix completion and result saving
    for LambdaIndex1 = 1:length(TempLambda1)
        lambda1 = TempLambda1(LambdaIndex1);
        for LambdaIndex2 = 1:length(TempLambda2)
            lambda2 = TempLambda2(LambdaIndex2);
            
            [X, history, min_error] = HCPIMSC(X1, O1, X2, O2, lambda1, lambda2, unfindIndex, c,fileIndex,num);
            
            if min_error < perfect_pross
                perfect_pross = min_error;
                pmin_error = min_error;
                l1 = lambda1;
                l2 = lambda2;
                
                % Save completed matrix X to a text file
                %fid = fopen(outputFileName, 'w');
                completeMatrix = [];
                
                for v = 1:length(X)
                    completeMatrix = [completeMatrix; X{v}];
                end
                
                % Save completeMatrix to CSV file
                writematrix(completeMatrix, csvOutputFileName);
                disp(['Matrix X saved to CSV file: ', csvOutputFileName]);
                
                % Write matrices to text file with view labels
                %for v = 1:length(X)
                    %fprintf(fid, 'View %d (Lambda %.3f):\n', v, lambda1);
                    %[m, n] = size(X{v});
                    %for i = 1:m
                     %   fprintf(fid, '%*s', -1, '');
                      %  fprintf(fid, '%.4f\t', X{v}(i, :));
                       % fprintf(fid, '\n');
                    %end
                    %fprintf(fid, '\n');
                %end
                
                %fclose(fid);
                %disp(['Saved one file for file index ', num2str(fileIndex)]);
            end
        end
    end
    fprintf('Perfect process Reconstruction error = %.6f for file index %d, Lambda1=%f, Lambda2=%f\n', pmin_error, fileIndex, l1, l2);
    %disp(['Combined matrices saved to ', outputFileName]);
end
