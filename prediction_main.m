% Prediction of multi-dimensional time-series data using different methods: RNNs trained with UORO and RTRL, LMS and linear regression.
% The performance is evaluated with the assumption that the data represents the 3D position of multiple objects.
% The data provided consists of the 3D position of external markers on the chest used during the radiotherapy treatment to accurately deliver radiation.
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

clear all
close all
clc

%% PARAMETERS 

% Program behavior
beh_par = load_behavior_parameters();

% Directories 
path_par = load_path_parameters();

% Display parameters
disp_par = load_display_parameters(path_par);

%% ---------------------------------------------------------------------------------------------------------------------------------------------------
%  PROGRAM -------------------------------------------------------------------------------------------------------------------------------------------
%  --------------------------------------------------------------------------------------------------------------------------------------------------- 

nb_seq = length(path_par.time_series_dir_tab);
for seq_idx = 1:nb_seq

    path_par.time_series_dir = path_par.time_series_dir_tab(seq_idx);
    path_par.input_seq_dir = sprintf('%s\\%s', path_par.parent_seq_dir, path_par.time_series_dir_tab(seq_idx));
    path_par.time_series_data_filename = sprintf('%s\\%s', path_par.input_seq_dir, path_par.time_series_data_filename_suffix);
    
    % Parameters concerning the prediction of the position of objects
    pred_par = load_pred_par(path_par);
    pred_par.t_eval_start = 1 + pred_par.tmax_cv; % car je veux faire l'eval sur l'ensemble de test
    pred_par.nb_predictions = pred_par.tmax_pred - pred_par.t_eval_start + 1;
    
    if beh_par.TRAIN_AND_PREDICT
        [Ypred, avg_pred_time, pred_loss_function] = train_and_predict(path_par, pred_par, beh_par);
    end

    if (beh_par.SAVE_PREDICTION_PLOT)||(beh_par.EVALUATE_PREDICTION)
        eval_results = pred_eval(beh_par, path_par, pred_par, disp_par, Ypred, avg_pred_time, pred_loss_function);
    end

    write_time_series_pred_log_file(path_par, beh_par, pred_par, eval_results);    
    
end