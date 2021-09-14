% Prediction of multi-dimensional time-series data using different methods: RNNs trained with UORO and RTRL, LMS and linear regression.
% The performance is evaluated with the assumption that the data represents the 3D position of multiple objects.
% The data provided consists of the 3D position of external markers on the chest used during the radiotherapy treatment to accurately deliver radiation.
%
% In this function, the hyperparameters relative to each prediction method are optimized using grid search
% 
% Matlab's parallel computing toolbox is needed to do parallel processing, which significantly reduces the computing time of grid search.
% To run the program without performing parallel computing, one can replace the "parfor" instructions with "for" instructions.
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

clear all
close all
clc

%% PARAMETERS

% GPU computing or not
beh_par.GPU_COMPUTING = false;

% Directories 
path_par = load_path_parameters();

% Display parameters
disp_par = load_display_parameters(path_par);  
               
nb_seq = length(path_par.time_series_dir_tab);
for seq_idx = 1:nb_seq

    % filename of the sequence being studied
    path_par.time_series_dir = path_par.time_series_dir_tab(seq_idx);
    path_par.input_seq_dir = sprintf('%s\\%s', path_par.parent_seq_dir, path_par.time_series_dir);
    path_par.time_series_data_filename = sprintf('%s\\%s', path_par.input_seq_dir, path_par.time_series_data_filename_suffix);    

    % Parameters concerning the prediction of the position of objects
    pred_par = load_pred_par(path_par);
    % Hyperparameters to optimize 
    hppars = load_hyperpar_cv_info( pred_par );
    
    %% PROGRAM

    [optim, best_par, par_influence] = train_eval_predictor_mult_param(hppars, pred_par, path_par, disp_par, beh_par);
    write_hppar_optim_log_file(hppars, pred_par, path_par, optim, best_par, par_influence, beh_par);
    
end  