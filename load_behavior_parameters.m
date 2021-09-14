function [ beh_par ] = load_behavior_parameters()
% The structure beh_par contains important information about the behavior of the whole algorithm.
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

%% IMPORTANT PARAMETERS
beh_par.TRAIN_AND_PREDICT = true;
    % if set to true, the RNN/predictor will be trained and predict future data.
beh_par.EVALUATE_PREDICTION = true;
    % if set to true, the predictor performance will be evaluated on the test set.
beh_par.SAVE_PREDICTION_PLOT = true;
    % if set to true, the predicted positions of the objects as well as the error loss function will be saved.
    
beh_par.SAVE_PRED_RESULTS = false;    
    % if set to true, the graphs containing the predicted objects' positions will be saved.

beh_par.GPU_COMPUTING = false;
    % increases computation speed with RTRL
    % NVidia GPU processor and Matlab's parallel computing toolbox is required
    
end