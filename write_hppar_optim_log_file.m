function write_hppar_optim_log_file( hppars, pred_par, path_par, optim, best_par, par_influence, beh_par)
% Writes the log file containing the results relative to selection of the best hyper-parameters using grid search
% as well as the influence of each parameter on the prediction accuracy.
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    log_file_entire_fname = sprintf('%s\\%s %s %s', path_par.txt_file_dir, path_par.time_series_dir, pred_par.pred_meth_str, path_par.hyperpar_optim_log_filename);
    fid = fopen(log_file_entire_fname,'wt');
    
    %% GENERAL PARAMETERS
    
    fprintf(fid, 'sequence name : %s \n', path_par.time_series_dir);
    fprintf(fid, '%s \n',path_par.date_and_time);

    % calculation paremeters        
    fprintf(fid, 'Prediction method : %s \n', pred_par.pred_meth_str);    
    fprintf(fid, 'Training between t = 1 and t = %d \n', pred_par.tmax_training);
    fprintf(fid, 'Cross-validation between t = %d and t = %d \n', 1+pred_par.tmax_training, pred_par.tmax_cv);
    
    if pred_par.NORMALIZE_DATA
        fprintf(fid, 'Data normalized before prediction \n');
    else
        fprintf(fid, 'Data not normalized before prediction \n');
    end
    
    fprintf(fid, 'Number of runs on the cross validation set due to random weights initialization nb_runs_cv = %d \n', hppars.nb_runs_cv);   
    fprintf(fid, 'Number of runs on the test set due to random weights initialization nb_runs_test = %d \n', hppars.nb_runs_eval_test);        

    switch(pred_par.pred_meth_idx)
        case {2,4,5}
                if pred_par.GRAD_CLIPPING % gradient clipping
                    fprintf(fid, 'Gradient clipping with threshold grd_tshld = %f \n', pred_par.grad_threshold);
                else
                    fprintf(fid, 'No gradient clipping \n');
                end
                switch pred_par.update_meth 
                    case 1
                        fprintf(fid, 'Optimization : stochastic gradient descent \n');
                    case 2
                        fprintf(fid, 'Optimization : ADAM  \n');
                end 
        otherwise 
    end
    
    fprintf(fid, 'Range of parameters tested \n');   
    fprintf(fid, 'Prediction horizon : \n');
    for horizon_idx = 1:hppars.nb_hrz_val
       fprintf(fid, '%d \t', hppars.horizon_tab(horizon_idx)); 
    end
    fprintf(fid, '\n');      

    for hppar_idx = 1:hppars.nb_additional_params
        fprintf(fid, '%s : \n', hppars.other(hppar_idx).name);
        for par_val_idx = 1:hppars.other(hppar_idx).nb_val
            fprintf(fid, '%g \t', hppars.other(hppar_idx).val(par_val_idx)); 
        end
        fprintf(fid, '\n');
    end
 
    switch pred_par.pred_meth_idx
        case {2, 4, 5} % RNN-RTRL, RNN-UORO, or LMS
            if (pred_par.update_meth == 2) % gradient clipping
                fprintf(fid, 'Gradient clipping with threshold grd_tshld = %f \n', pred_par.grad_threshold);
            end
    end
    
    fprintf(fid, '\n \n');
    
    
    %% BEST PARAMETERS AND INFLUENCE OF THE PREDICTION HORIZON  

    
    for horizon_idx = 1:hppars.nb_hrz_val
    
        fprintf(fid, 'Best performance found for the horizon h = %d \n', hppars.horizon_tab(horizon_idx)); 
        fprintf(fid, 'Nb explosions on the cross validation set : %d \n', best_par.nb_expl_cv_tab(horizon_idx));
        fprintf(fid, 'average time for predicting the data at t+1 from the past data : %f s \n', best_par.mean_pt_pos_pred_time_tab(horizon_idx));
        
        switch(pred_par.pred_meth_idx)
        
            case {1,3,4} % multivariate linear regression or no prediction or clipped multivar LMS (deterministic algorithms)
        
                fprintf(fid, 'mean prediction error on the test set: %f mm \n', best_par.mean_err_test_set_tab(horizon_idx));
                fprintf(fid, '(mean of the) max prediction error on the test set : %f mm \n', best_par.max_err_test_set_tab(horizon_idx));
                fprintf(fid, '(mean of the) rms error on the test set : %f mm \n', best_par.rms_err_test_set_tab(horizon_idx));
                fprintf(fid, '(mean of the) jitter on the test set : %f mm \n', best_par.jitter_test_set_tab(horizon_idx));
                fprintf(fid, '(mean of the) nmrse on the test set : %f \n', best_par.nrmse_test_set_tab(horizon_idx));                
                
                fprintf(fid, 'Same info but column for copy-paste in excel file (mean - rms - max - jitter on test set with half confidence) : \n');  
                fprintf(fid, '%f \n', best_par.mean_err_test_set_tab(horizon_idx));
                fprintf(fid, '%f \n', best_par.rms_err_test_set_tab(horizon_idx));
                fprintf(fid, '%f \n', best_par.max_err_test_set_tab(horizon_idx));
                fprintf(fid, '%f \n', best_par.jitter_test_set_tab(horizon_idx));
                fprintf(fid, '%f \n', best_par.nrmse_test_set_tab(horizon_idx));                
                
            otherwise
                
                fprintf(fid, 'mean prediction error on the test set: %f mm \n', best_par.mean_err_test_set_tab(horizon_idx));
                fprintf(fid, 'mean prediction error 95%% half confidence interval on the test set : %f (mm) \n', best_par.cf_half_range_mean_err_test_set_tab(horizon_idx));

                fprintf(fid, '(mean of the) max prediction error on the test set : %f mm \n', best_par.max_err_test_set_tab(horizon_idx));
                fprintf(fid, 'max prediction error 95%% half confidence interval on the test set : %f (mm) \n', best_par.cf_half_range_max_err_test_set_tab(horizon_idx));

                fprintf(fid, '(mean of the) rms error on the test set : %f mm \n', best_par.rms_err_test_set_tab(horizon_idx));
                fprintf(fid, 'rms prediction error 95%% half confidence interval on the test set : %f (mm) \n', best_par.cf_half_range_rms_err_test_set_tab(horizon_idx)); 

                fprintf(fid, '(mean of the) jitter on the test set : %f mm \n', best_par.jitter_test_set_tab(horizon_idx));
                fprintf(fid, 'jitter 95%% half confidence interval on the test set : %f (mm) \n', best_par.cf_half_range_jitter_test_set_tab(horizon_idx));  
                
                fprintf(fid, '(mean of the) nmrse on the test set : %f \n', best_par.nrmse_test_set_tab(horizon_idx));
                fprintf(fid, 'nrmse 95%% half confidence interval on the test set : %f (mm) \n', best_par.cf_half_range_nrmse_test_set_tab(horizon_idx));                 

                fprintf(fid, 'Same info but column for copy-paste in excel file (mean - rms - max - jitter on test set with half confidence) : \n');  
                fprintf(fid, '%f \n', best_par.mean_err_test_set_tab(horizon_idx));
                fprintf(fid, '%f \n', best_par.cf_half_range_mean_err_test_set_tab(horizon_idx));
                fprintf(fid, '%f \n', best_par.rms_err_test_set_tab(horizon_idx));
                fprintf(fid, '%f \n', best_par.cf_half_range_rms_err_test_set_tab(horizon_idx));  
                fprintf(fid, '%f \n', best_par.max_err_test_set_tab(horizon_idx));
                fprintf(fid, '%f \n', best_par.cf_half_range_max_err_test_set_tab(horizon_idx));
                fprintf(fid, '%f \n', best_par.jitter_test_set_tab(horizon_idx));
                fprintf(fid, '%f \n', best_par.cf_half_range_jitter_test_set_tab(horizon_idx));  
                fprintf(fid, '%f \n', best_par.nrmse_test_set_tab(horizon_idx));
                fprintf(fid, '%f \n', best_par.cf_half_range_nrmse_test_set_tab(horizon_idx));                
        
        end

        fprintf(fid, 'Corresponding parameters : \n');     
        for hppar_idx = 1:hppars.nb_additional_params
            fprintf(fid, '%s = %g \n', hppars.other(hppar_idx).name, best_par.other_hyppar_tab(horizon_idx, hppar_idx));
        end
        fprintf(fid, '\n');
    
    end
    
    
    %% NUMBER OF NUMERICAL ERRORS
    
    total_num_errors = 0;
    for hrz_idx = 1:hppars.nb_hrz_val
        total_num_errors = total_num_errors + my_sum(optim(hrz_idx).nb_xplosion_tab);
    end
    fprintf(fid, 'Total number of numerical errors : %d \n', total_num_errors);
    fprintf(fid, 'Total number of calculations : %d \n', hppars.nb_calc*hppars.nb_runs_eval_test);    
    fprintf(fid, 'Ratio numerical errors / calculations : %g \n', total_num_errors/(hppars.nb_calc*hppars.nb_runs_eval_test));
    fprintf(fid, '\n');
    

    %% INFLUENCE OF EACH PARAMETERS

    if beh_par.GPU_COMPUTING
       fprintf(fid, 'Computations with the GPU \n');
    else
       fprintf(fid, 'Computations with the CPU \n');        
    end
    
    switch(pred_par.pred_meth_idx)
        case {2,5} % RNN-RTRL or RNN-UORO
            fprintf(fid, 'Average prediction time as a function of the SHL (line) and the number of neurons in the hidden layer (columns) in s \n');
        otherwise
            fprintf(fid, 'Average prediction time as a function of the SHL in s \n');
    end
    for ii = 1:size(par_influence.pred_time_avg,1)
        fprintf(fid,'%g\t',par_influence.pred_time_avg(ii,:));
        fprintf(fid,'\n');
    end    
    fprintf(fid, '\n');     
    
    if (pred_par.pred_meth_idx ~= 3)    
        fprintf(fid, 'Minimum of the nRMSE on the cross-validation set for each hyper-parameter as a function of the horizon : \n');
        fprintf(fid, 'Lines : hyper-parameters / Column : horizon values \n');
        for hppar_idx = 1:hppars.nb_additional_params
            fprintf(fid, 'Influence of %s : \n', hppars.other(hppar_idx).name);
            for hppar_val_idx = 1:hppars.other(hppar_idx).nb_val
                for hrz_idx = 1:hppars.nb_hrz_val
                    fprintf(fid,'%8f \t',par_influence.min_nRMSE{hppar_idx,hrz_idx}(hppar_val_idx));
                end
                fprintf(fid,'\n');
            end
        end    
    end    
    fprintf(fid, '\n');       
 
    fclose(fid);
        
end
