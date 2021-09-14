function [optim, best_par, par_influence] = train_eval_predictor_mult_param(hppars, pred_par, path_par, disp_par, beh_par)
% Performs the training and evaluation of the prediction method selected in load_pred_par
% using grid search with the hyperparameter grid selected in load_hyperpar_cv.
% The influence of each hyper-parameter on the prediction result is evaluated.
% Parallel processing is used to accelerate the speed of grid search.
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

fprintf('Optimization of the parameters of the prediction algorithm \n\n');
        
    beh_par.SAVE_PREDICTION_PLOT = false;
    beh_par.SAVE_PRED_RESULTS = false;  
    
        
    %% EVALUATION ON ALL THE PARAMETERS
 
    % amount of data for training, for cross validation and for testing
    pred_par.t_eval_start = 1 + pred_par.tmax_training;
    pred_par.nb_predictions = pred_par.tmax_cv - pred_par.t_eval_start + 1;
    pred_par.tmax_pred = pred_par.tmax_cv;
    pred_par.nb_runs = hppars.nb_runs_cv;    
    
    size_other_hyppr_tab = [];
    for hppar_idx = 1:hppars.nb_additional_params
        size_other_hyppr_tab = [size_other_hyppr_tab, hppars.other(hppar_idx).nb_val];
    end
    if (hppars.nb_additional_params == 1)
        size_other_hyppr_tab = [size_other_hyppr_tab, 1];
    end
    
    optim = struct();
    for hrz_idx = 1:hppars.nb_hrz_val
        optim(hrz_idx).rms_error_tab = zeros(size_other_hyppr_tab, 'single');
        optim(hrz_idx).rms_error_confidence_half_range_tab = zeros(size_other_hyppr_tab, 'single');
        optim(hrz_idx).nb_xplosion_tab = zeros(size_other_hyppr_tab, 'single');
        optim(hrz_idx).nrmse_tab = zeros(size_other_hyppr_tab, 'single');
        optim(hrz_idx).pred_time_tab = zeros(size_other_hyppr_tab, 'single');
        optim(hrz_idx).jitter_tab = zeros(size_other_hyppr_tab, 'single');      
    end
    
    
    parfor hrz_idx = 1:hppars.nb_hrz_val
        
        pred_par_h = pred_par;
        pred_par_h.horizon = hppars.horizon_tab(hrz_idx);
        
        v_h = ones(1, hppars.nb_additional_params);
        nb_calc_crt = 1;     
        
        ready = false;
        optim(hrz_idx) = perform_cv_once( v_h, optim(hrz_idx), nb_calc_crt, hppars, pred_par_h, path_par, beh_par, disp_par);
        while ~ready
            % Update the index vector:
            ready = true;
            for k = hppars.nb_additional_params:-1:1
                v_h(k) = v_h(k) + 1;
                if v_h(k) <= size_other_hyppr_tab(k)

                    ready = false; 
                    nb_calc_crt = nb_calc_crt +1;
                    optim(hrz_idx) = perform_cv_once( v_h, optim(hrz_idx), nb_calc_crt, hppars, pred_par_h, path_par, beh_par, disp_par);                            
                    
                    break;  % v(k) increased successfully, leave the "for k" loop

                end
                v_h(k) = 1;  % v(k) reached the limit, reset it and iterate v(k-1)
            end
        end   
  
    end
    
    %% SEARCH FOR THE BEST PARAMETERS
    
    beh_par.SAVE_PREDICTION_PLOT = false;
        
    best_par.rms_cv_error_tab = zeros(hppars.nb_hrz_val, 1, 'single');
    best_par.nb_expl_cv_tab = zeros(hppars.nb_hrz_val, 1, 'single');
    
    best_par.other_hyppar_tab = zeros(hppars.nb_hrz_val, hppars.nb_additional_params, 'single');
    
    best_par.mean_err_test_set_tab = zeros(hppars.nb_hrz_val, 1, 'single');
    best_par.cf_half_range_mean_err_test_set_tab = zeros(hppars.nb_hrz_val, 1, 'single');
    best_par.rms_err_test_set_tab = zeros(hppars.nb_hrz_val, 1, 'single');
    best_par.cf_half_range_rms_err_test_set_tab = zeros(hppars.nb_hrz_val, 1, 'single');
    best_par.max_err_test_set_tab = zeros(hppars.nb_hrz_val, 1, 'single');
    best_par.cf_half_range_max_err_test_set_tab = zeros(hppars.nb_hrz_val, 1, 'single');
    best_par.jitter_test_set_tab = zeros(hppars.nb_hrz_val, 1, 'single');
    best_par.cf_half_range_jitter_test_set_tab = zeros(hppars.nb_hrz_val, 1, 'single');
    best_par.nrmse_test_set_tab = zeros(hppars.nb_hrz_val, 1, 'single');
    best_par.cf_half_range_nrmse_test_set_tab = zeros(hppars.nb_hrz_val, 1, 'single');    
    best_par.mean_pt_pos_pred_time_tab = zeros(hppars.nb_hrz_val, 1, 'single');
    best_par.nb_xplosions_tab = zeros(hppars.nb_hrz_val, 1, 'single');
    
    pred_par_cell = cell(hppars.nb_hrz_val, 1);
    	% eval_results_best_par is a cell which contains structures
    
    for hrz_idx = 1:hppars.nb_hrz_val
        
        error_aux_tab = optim(hrz_idx).rms_error_tab;
            % error_aux_tab contains the RMS values of the prediction algorithm for the considered hyperparameters and the current horizon value
        nb_xplosion_tab_temp = optim(hrz_idx).nb_xplosion_tab;
        
        min_expl = my_min(nb_xplosion_tab_temp);
        nb_xplosion_tab_temp = nb_xplosion_tab_temp - min_expl*ones(size(nb_xplosion_tab_temp));
        expl_idx_tab = nb_xplosion_tab_temp~=0;
        error_aux_tab(expl_idx_tab) = Inf;
            % parameters for which there are numerical errors are not taken into account
        best_par.rms_cv_error_tab(hrz_idx) = my_min(error_aux_tab);
        lin_idx_min = find(error_aux_tab == best_par.rms_cv_error_tab(hrz_idx));
        
        idx_vec = my_ind2sub(size(error_aux_tab), lin_idx_min);
        
        best_par.nb_expl_cv_tab(hrz_idx) = min_expl;
        for hppar_idx = 1:hppars.nb_additional_params
            best_par.other_hyppar_tab(hrz_idx, hppar_idx) = hppars.other(hppar_idx).val(idx_vec(hppar_idx));
        end

        % evaluation on the test set with the optimal parameters
        pred_par_cell{hrz_idx} = load_pred_par(path_par); %we find tmax_pred
            % in the case of linear regression, the value of pred_par.tmax_training is already modified inside the function load_pred_par.m
        pred_par_cell{hrz_idx}.t_eval_start = 1 + pred_par_cell{hrz_idx}.tmax_cv; % because evaluation on the test set
        pred_par_cell{hrz_idx}.nb_predictions = pred_par_cell{hrz_idx}.tmax_pred - pred_par_cell{hrz_idx}.t_eval_start + 1;
        pred_par_cell{hrz_idx}.nb_runs = hppars.nb_runs_eval_test;    
        
        pred_par_cell{hrz_idx}.horizon = hppars.horizon_tab(hrz_idx);
        for hppar_idx = 1:hppars.nb_additional_params
            pred_par_cell{hrz_idx}.(hppars.other(hppar_idx).name) = best_par.other_hyppar_tab(hrz_idx, hppar_idx);
        end
 
    end
        
    parfor hrz_idx = 1:hppars.nb_hrz_val   
 
        fprintf('\n \n');
        fprintf('Eval on test data for h = %d \n', hppars.horizon_tab(hrz_idx))        
        
        [Ypred, avg_pred_time, pred_loss_function] = train_and_predict(path_par, pred_par_cell{hrz_idx}, beh_par);
        eval_results_best_par(hrz_idx) = pred_eval(beh_par, path_par, pred_par_cell{hrz_idx}, disp_par, Ypred, avg_pred_time, pred_loss_function);   
            % eval_results_best_par is a structure array
    
    end
        
    for hrz_idx = 1:hppars.nb_hrz_val    
        
        best_par.mean_err_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).mean_mean_err;
        best_par.cf_half_range_mean_err_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).confidence_half_range_mean_err;
        best_par.rms_err_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).mean_rms_err;
        best_par.cf_half_range_rms_err_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).confidence_half_range_rms_err;
        best_par.max_err_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).mean_max_err;
        best_par.cf_half_range_max_err_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).confidence_half_range_max_err;
        best_par.jitter_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).mean_jitter;
        best_par.cf_half_range_jitter_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).confidence_half_range_jitter;
        best_par.nrmse_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).mean_nrmse;
        best_par.cf_half_range_nrmse_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).confidence_half_range_nrmse;
        best_par.mean_pt_pos_pred_time_tab(hrz_idx) = eval_results_best_par(hrz_idx).mean_pt_pos_pred_time;
        best_par.nb_xplosions_tab(hrz_idx) = eval_results_best_par(hrz_idx).nb_xplosion;
        
    end

    %% STUDY OF THE INFLUENCE OF EACH PARAMETERS
    
    % Average prediction time to make one prediction (Columns : number of neurons in the hidden layer / lines : SHL)
    temp_avg_time = 0; 
        % broadcasting here because the size of temp_avg_time depends on pred_par.pred_meth_idx
    for hrz_idx = 1:hppars.nb_hrz_val
        pred_time_tab = optim(hrz_idx).pred_time_tab;
        for hppar_idx = 1:hppars.nb_additional_params
            switch(pred_par.pred_meth_idx)
                case {2,5} % RNN-RTRL or RNN-UORO
                    if (hppar_idx ~= hppars.state_space_hyppar_idx)&&(hppar_idx ~= hppars.SHL_hyppar_idx)
                            % we want to study the influence of the number of hidden neurons and SHL so we do not compute the mean over these variables 
                        pred_time_tab = mean(pred_time_tab, hppar_idx); 
                    end
                otherwise
                    if (hppar_idx ~= hppars.SHL_hyppar_idx)
                            % we want to study the influence of the SHL so we do not compute the mean over it
                        pred_time_tab = mean(pred_time_tab, hppar_idx);
                    end
            end 
        end
        temp_avg_time = temp_avg_time + pred_time_tab;
    end
    temp_avg_time = temp_avg_time/hppars.nb_hrz_val; % mean calculation
    par_influence.pred_time_avg = squeeze(temp_avg_time);
        
    if (hppars.nb_additional_params >=1) % we eliminate the case without prediction
        par_influence.min_nRMSE = cell(hppars.nb_additional_params, hppars.nb_hrz_val);
        if (hppars.nb_additional_params ==1) % typically linear regression
           for hrz_idx = 1:hppars.nb_hrz_val
               par_influence.min_nRMSE{1, hrz_idx} = optim(hrz_idx).nrmse_tab;
                    % minimum of the nRMSE over the cross validation set
           end
        else
            for hppar_idx = 1:hppars.nb_additional_params
               for hrz_idx = 1:hppars.nb_hrz_val
                   vecdim = 1:hppars.nb_additional_params;
                   vecdim(hppar_idx) = [];
                   par_influence.min_nRMSE{hppar_idx, hrz_idx} = min(optim(hrz_idx).nrmse_tab, [], vecdim);
                        % minimum of the nRMSE over the cross validation set
               end
            end
        end
    end

end
