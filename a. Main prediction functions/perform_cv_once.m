function optim = perform_cv_once( v, optim, nb_calc_crt, hppars, pred_par, path_par, beh_par, disp_par)
% performs training and evaluation with the hyperparameters designated by the index vector v
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    fprintf('\n \n'); 
    fprintf('Sequence %s \n', path_par.time_series_dir);                            
    fprintf('Test (cross validation) %d/%d with the following parameters : \n', nb_calc_crt, hppars.nb_calc);
    fprintf('\t prediction horizon = %d \n', pred_par.horizon);   

    for hppar_idx = 1:hppars.nb_additional_params
        hppar_name = hppars.other(hppar_idx).name;
        pred_par.(hppar_name) = hppars.other(hppar_idx).val(v(hppar_idx));
        fprintf('\t %s = %g \n', hppar_name, pred_par.(hppar_name));
    end    
    fprintf('\n');

    [Ypred, avg_pred_time, pred_loss_function] = train_and_predict(path_par, pred_par, beh_par);
    eval_results = pred_eval(beh_par, path_par, pred_par, disp_par, Ypred, avg_pred_time, pred_loss_function);

    hppar_idx_cell = num2cell(v);
    optim.rms_error_tab(hppar_idx_cell{:}) = eval_results.mean_rms_err;
    optim.rms_error_confidence_half_range_tab(hppar_idx_cell{:}) = eval_results.confidence_half_range_rms_err;                
    optim.nb_xplosion_tab(hppar_idx_cell{:}) = eval_results.nb_xplosion;
    optim.nrmse_tab(hppar_idx_cell{:}) = eval_results.mean_nrmse;
    optim.pred_time_tab(hppar_idx_cell{:}) = eval_results.mean_pt_pos_pred_time;
    optim.jitter_tab(hppar_idx_cell{:}) = eval_results.mean_jitter;                             

end

