function [eval_results] = pred_eval(beh_par, path_par, pred_par, disp_par, Ypred, avg_pred_time, pred_loss_function)
% Evaluation of the prediction performance of the prediction methods
% We make the assumption that the data to be predicted consists of the position of 3D points.
% This function :
%   - computes the mean average prediction error, the root-mean square error (RMSE), the maximum error, the normalized RMSE, the jitter, 
%       the associated 95% mean confidence intervals, the number of numerical erros, and the prediction time, which are stored in the structure eval_results.
%   - displays and saves plots comparing the predicted and original positions
%   - displays and saves plots of the instantaneous error function.
% 
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    pt_pos_pred_time = zeros(pred_par.nb_runs, 1);
        % time for predicting the position of the markers at each time step
    eval_results = struct();
    eval_results.mean_err = zeros(pred_par.nb_runs, 1);
    eval_results.max_err = zeros(pred_par.nb_runs, 1);
    eval_results.rms_err = zeros(pred_par.nb_runs, 1);
    eval_results.rmse_err = zeros(pred_par.nb_runs, 1);
    eval_results.nrmse = zeros(pred_par.nb_runs, 1);
    eval_results.jitter = zeros(pred_par.nb_runs, 1);

    % Loading the true 3D position of the points
    load(path_par.time_series_data_filename, 'org_data');
    org_data = org_data(:,1:pred_par.tmax_pred);
    [data_dim, ~] = size(org_data);
    
    nb_pts = data_dim/3; % because we are predicting the 3D position of points.
    true_pts_pos = zeros(pred_par.tmax_pred, nb_pts, 3);

    org_data = org_data.';
    true_pts_pos(:,:,1) = org_data(:,1:nb_pts);                %x coordinates of the points 
    true_pts_pos(:,:,2) = org_data(:,(nb_pts+1):(2*nb_pts));   %y coordinates of the points 
    true_pts_pos(:,:,3) = org_data(:,(2*nb_pts+1):(3*nb_pts)); %z coordinates of the points 

    mu_true = zeros(3, nb_pts); % used for calculating the nrmse
    nrmse_denominator = zeros(pred_par.nb_predictions, nb_pts);
    ptwise_err_tab = zeros(pred_par.nb_predictions, nb_pts);
    RNNloss_all_runs = zeros(pred_par.tmax_pred, pred_par.nb_runs); 
    
    for run_idx=1:pred_par.nb_runs

        if (pred_par.nb_runs>1)&&(mod(run_idx,10) == 1)
            fprintf('Evaluation for the %d-th test (random initialization) \n', run_idx);
        end

        [pred_pts_pos, pred_calc_time, RNNloss ] = return_pred_results( pred_par, run_idx, Ypred, avg_pred_time, pred_loss_function );
        RNNloss_all_runs(:, run_idx) = RNNloss;
        pt_pos_pred_time(run_idx) = pred_calc_time;

        % Mean, RMS and maximum error calculation
        delta = true_pts_pos(pred_par.t_eval_start:pred_par.tmax_pred,:,:) - pred_pts_pos(pred_par.t_eval_start:pred_par.tmax_pred,:,:);
        for pt_idx = 1:nb_pts
            for t = 1:pred_par.nb_predictions
                ptwise_err_tab(t, pt_idx) = sqrt(delta(t,pt_idx, 1)^2 + delta(t,pt_idx, 2)^2 + delta(t,pt_idx, 3)^2 );
            end
        end
        eval_results.mean_err(run_idx) = (1/(nb_pts*pred_par.nb_predictions))*sum(sum(ptwise_err_tab));
        eval_results.max_err(run_idx) = max(max(ptwise_err_tab));
        eval_results.rms_err(run_idx) = (1/sqrt(nb_pts*pred_par.nb_predictions))*sqrt(sum(sum(ptwise_err_tab.^2)));  

        % NMRSE calculation
        true_pts_pos_test_permuted = permute(true_pts_pos(pred_par.t_eval_start:pred_par.tmax_pred,:,:), [3,2,1]); % coordinate - pt idx - time        
        for pt_idx = 1:nb_pts
            mu_true(:, pt_idx) = mean(true_pts_pos_test_permuted(:, pt_idx, :), 3); 
        end
        for pt_idx = 1:nb_pts
            for t = 1:pred_par.nb_predictions
                nrmse_denominator(t, pt_idx) = sqrt(sum((true_pts_pos_test_permuted(:,pt_idx,t) - mu_true(:, pt_idx)).^2, 1));
            end
        end        
        eval_results.nrmse(run_idx) = sqrt(sum(sum(ptwise_err_tab.^2)))/sqrt(sum(sum(nrmse_denominator.^2)));

        % Jitter calculation
        instant_jitter = sqrt(sum((pred_pts_pos((pred_par.t_eval_start+1):pred_par.tmax_pred,:,:)-pred_pts_pos(pred_par.t_eval_start:(pred_par.tmax_pred-1),:,:)).^2,3));
        eval_results.jitter(run_idx) = mean(mean(instant_jitter));

        SAVE_ONLY = (run_idx ~=1); % only the plots from the first test remain displayed on the screen.
        if (beh_par.SAVE_PREDICTION_PLOT)&&(run_idx <= disp_par.nb_pred_runs_saved)
            for pt_idx = 1:nb_pts
                plot_pred_coord( pred_pts_pos, true_pts_pos, pred_par, path_par, disp_par, pt_idx, 1, run_idx, SAVE_ONLY);
                plot_pred_coord( pred_pts_pos, true_pts_pos, pred_par, path_par, disp_par, pt_idx, 2, run_idx, SAVE_ONLY);
                plot_pred_coord( pred_pts_pos, true_pts_pos, pred_par, path_par, disp_par, pt_idx, 3, run_idx, SAVE_ONLY);
            end
                title_str = sprintf('Prediction loss function (run %d)', run_idx);
                filename_suffix = sprintf('%s pred loss function %s %d-th run', path_par.time_series_dir, sprintf_pred_param(pred_par), run_idx);
                plot_pred_error( RNNloss, disp_par, pred_par, path_par, filename_suffix, title_str, SAVE_ONLY );
        end
    
    end
    
    % counts the number of times when the prediction fails due to a numerical error (typically gradient explosion) 
    num_error_idx_vec = any(isnan(eval_results.rms_err),2);
    eval_results.nb_xplosion = sum(num_error_idx_vec); 
    eval_results.nb_correct_runs = pred_par.nb_runs - eval_results.nb_xplosion;
    RNNloss_all_runs(:, num_error_idx_vec) = [];
    
    % plot of the mean error function
    if (pred_par.pred_meth_idx==2)||(pred_par.pred_meth_idx==5)
        
        meanloss = mean(RNNloss_all_runs, 2);
            
        if (beh_par.SAVE_PREDICTION_PLOT)
            SAVE_ONLY = false;     
            filename_suffix = sprintf('%s pred mean loss function %s', path_par.time_series_dir, sprintf_pred_param(pred_par));
            title_str = sprintf('Prediction mean loss function');
            plot_pred_error( meanloss, disp_par, pred_par, path_par, filename_suffix, title_str, SAVE_ONLY )
        end
    end

    % removing the NaN rows (numerical error) in the evaluation process
    pred_time_temp_mat = pt_pos_pred_time;
    pred_time_temp_mat(num_error_idx_vec) = [];
    pred_mean_err_temp_mat = eval_results.mean_err;
    pred_mean_err_temp_mat(num_error_idx_vec) = [];
    pred_max_err_temp_mat = eval_results.max_err;
    pred_max_err_temp_mat(num_error_idx_vec) = [];
    pred_rms_err_temp_mat = eval_results.rms_err;
    pred_rms_err_temp_mat(num_error_idx_vec) = [];  
    pred_nrmse_temp_mat = eval_results.nrmse;
    pred_nrmse_temp_mat(num_error_idx_vec) = [];      
    pred_jitter_temp_mat = eval_results.jitter;
    pred_jitter_temp_mat(num_error_idx_vec) = []; 
    
    % calculating evaluation statistics  
    eval_results.mean_pt_pos_pred_time = mean(pred_time_temp_mat);
    eval_results.mean_mean_err = mean(pred_mean_err_temp_mat);
    eval_results.confidence_half_range_mean_err = 1.96*std(pred_mean_err_temp_mat)/sqrt(eval_results.nb_correct_runs);
    eval_results.mean_max_err = mean(pred_max_err_temp_mat);
    eval_results.confidence_half_range_max_err = 1.96*std(pred_max_err_temp_mat)/sqrt(eval_results.nb_correct_runs);  
    eval_results.mean_rms_err = mean(pred_rms_err_temp_mat);
    eval_results.confidence_half_range_rms_err = 1.96*std(pred_rms_err_temp_mat)/sqrt(eval_results.nb_correct_runs); 
    eval_results.mean_nrmse = mean(pred_nrmse_temp_mat);    
    eval_results.confidence_half_range_nrmse = 1.96*std(pred_nrmse_temp_mat)/sqrt(eval_results.nb_correct_runs); 
    eval_results.mean_jitter = mean(pred_jitter_temp_mat);
    eval_results.confidence_half_range_jitter = 1.96*std(pred_jitter_temp_mat)/sqrt(eval_results.nb_correct_runs); 
    
end

