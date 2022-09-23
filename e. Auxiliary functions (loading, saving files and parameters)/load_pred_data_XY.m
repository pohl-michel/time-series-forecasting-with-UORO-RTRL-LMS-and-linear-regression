function [ X, Y, Mu, Sg] = load_pred_data_XY( path_par, pred_par, beh_par)
% Loading the past data matrix X and the future data matrix Y from the original data file.
%
% Example with the prediction of 2 signals of dimension 3
% the original data matrix would have the following form
%            [ x_1(t_1), ..., x_1(t_M+k+h-1)]
%            [ x_2(t_1), ..., x_2(t_M+k+h-1)]
% data     = [ y_1(t_1), ..., y_1(t_M+k+h-1)]
%            [ y_2(t_1), ..., y_2(t_M+k+h-1)]
%            [ z_1(t_1), ..., z_1(t_M+k+h-1)]
%            [ z_2(t_1), ..., z_2(t_M+k+h-1)]
% with k being the signal history length
% h the prediction horizon
% M the number of predictions
% and M+k+h-1 = pred_par.tmax_pred.
%
% The data matrices X and Y would then be :
%         [ 1       , ..., 1           ]
%         [ x_1(t_1), ..., x_1(t_M)    ]
%         [ x_2(t_1), ..., x_2(t_M)    ]
%         [ y_1(t_1), ..., y_1(t_M)    ]
%         [ y_2(t_1), ..., y_2(t_M)    ]
%         [ z_1(t_1), ..., z_1(t_M)    ]
%         [ z_2(t_1), ..., z_2(t_M)    ]
% Xdata = [           ...              ]
%         [ x_1(t_k), ..., x_1(t_M+k-1)]
%         [ x_2(t_k), ..., x_2(t_M+k-1)]
%         [ y_1(t_k), ..., y_1(t_M+k-1)]
%         [ y_2(t_k), ..., y_2(t_M+k-1)]
%         [ z_1(t_k), ..., z_1(t_M+k-1)]
%         [ z_2(t_k), ..., z_2(t_M+k-1)]
%
%         [ x_1(t_k+h), ..., x_1(t_M+k+h-1)]
%         [ x_2(t_k+h), ..., x_2(t_M+k+h-1)]
% Ydata = [ y_1(t_k+h), ..., y_1(t_M+k+h-1)]
%         [ y_2(t_k+h), ..., y_2(t_M+k+h-1)]
%         [ z_1(t_k+h), ..., z_1(t_M+k+h-1)]
%         [ z_2(t_k+h), ..., z_2(t_M+k+h-1)]
%
% Each variable x_i, y_i or z_i is standardized, ie in Xdata and Ydata the variables x_i' = [x_i - E(x_i)]/sqrt(V(x_i)) are used instead of x_i
% This results in a better stability of the overall RNN and faster learning
% 
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License


    load(path_par.time_series_data_filename, 'org_data');
    org_data = org_data(:,1:pred_par.tmax_pred);
        % if tmax_pred < the total amount of time steps in the variable 'org_data' then we crop that variable to decrease the calculation time
    [data_dim, ~] = size(org_data);
    k = pred_par.SHL; % k is the nb of successive time steps to perform one prediction
    h = pred_par.horizon; 
    M = pred_par.tmax_pred-k-h+1; % "number of predictions"
    
    % X matrix initialization
    X = ones(1+k*data_dim, M); 
    
    if pred_par.NORMALIZE_DATA
        % The TRAINING data is used to calculate the mean and standard deviation of each of the variables x1, x2, etc.
        % These statistics are used to normalize the whole data
        [ ~, Mu, Sg ] = standard_scores( org_data(:,1:pred_par.tmax_training));
        org_data = normalize_from_computed_stats( org_data, Mu, Sg);
    else
        Mu = 0;
        Sg = 0;
    end
        
    for j =1:M
        Ttemp = org_data(:,j:(j+k-1)); % size (1+k*data_dim, k)
        X(2:(1+k*data_dim),j) = Ttemp(:);
    end
   
    Y = double(org_data(:, (k+h):(M+k+h-1)));
    
    if beh_par.GPU_COMPUTING
        X = gpuArray(X);
        Y = gpuArray(Y);
    end

end