function write_time_series_pred_log_file(path_par, beh_par, pred_par, eval_results)
% Records the parameters used and the prediction numerical results in a txt file.
% 
% Author : Pohl Michel
% Date : August 12th, 2020
% Version : v1.0
% License : 3-clause BSD License

    log_file_complete_filename = sprintf('%s\\%s %s %s', path_par.txt_file_dir, path_par.time_series_dir, pred_par.pred_meth_str, path_par.log_txt_filename);
    fid = fopen(log_file_complete_filename,'wt');
        
        fprintf(fid, 'sequence name : %s \n', path_par.input_seq_dir);
        fprintf(fid, '%s \n',path_par.date_and_time);
 
        % I] Recording the calculation paremeters        

        fprintfpred_par( fid, pred_par, beh_par );
        fprintf(fid, '\n');
        
        % II] Recording the evaluation results
        if beh_par.EVALUATE_PREDICTION
        
            fprintf(fid, 'Calculation time \n');
            fprintf(fid, 'Average time for predicting the position at t+%d given the data until t : %e s\n', pred_par.horizon, eval_results.mean_pt_pos_pred_time);
            fprintf(fid, '\n');

            fprintf(fid, 'Evaluation results \n');
            fprintf(fid, 'nb of prediction runs with numerical error (gradient explosion) : %d \n', eval_results.nb_xplosion);
            
            fprintf(fid, 'mean prediction error : %f mm \n', eval_results.mean_mean_err);
            fprintf(fid, 'mean prediction error 95%% confidence half range : %f (mm) \n', eval_results.confidence_half_range_mean_err);            
            fprintf(fid, '(mean of the) rms error : %f mm \n', eval_results.mean_rms_err);
            fprintf(fid, 'rms prediction error 95%% confidence half range : %f (mm) \n', eval_results.confidence_half_range_rms_err);  
            fprintf(fid, '(mean of the) max prediction error : %f mm \n', eval_results.mean_max_err);
            fprintf(fid, 'max prediction error 95%% confidence half range : %f (mm) \n', eval_results.confidence_half_range_max_err);            
            fprintf(fid, '(mean of the) jitter : %f mm \n', eval_results.mean_jitter);
            fprintf(fid, 'jitter 95%% confidence half range : %f (mm) \n', eval_results.confidence_half_range_jitter);    
            fprintf(fid, '(mean of the) NRMSE : %f \n', eval_results.mean_nrmse);
            fprintf(fid, 'NRMSE 95%% confidence half range : %f \n', eval_results.confidence_half_range_nrmse);            
            fprintf(fid, 'Same info but column for copy-paste in excel file (mean - rms - max - jitter on test set with half confidence) : \n');  
            fprintf(fid, '%f \n', eval_results.mean_mean_err);
            fprintf(fid, '%f \n', eval_results.confidence_half_range_mean_err);            
            fprintf(fid, '%f \n', eval_results.mean_rms_err);
            fprintf(fid, '%f \n', eval_results.confidence_half_range_rms_err);            
            fprintf(fid, '%f \n', eval_results.mean_max_err); 
            fprintf(fid, '%f \n', eval_results.confidence_half_range_max_err);             
            fprintf(fid, '%f \n', eval_results.mean_jitter); 
            fprintf(fid, '%f \n', eval_results.confidence_half_range_jitter);  
            fprintf(fid, '%f \n', eval_results.mean_nrmse); 
            fprintf(fid, '%f \n', eval_results.confidence_half_range_nrmse);        
                    
        end 

        fprintf(fid, '\n');
        
    fclose(fid);

end