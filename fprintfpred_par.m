function fprintfpred_par( fid, pred_par, beh_par )
% Prints the prediction parameters relative to each prediction method.
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License
    
        fprintf(fid, 'Prediction method : %s \n', pred_par.pred_meth_str);
        fprintf(fid, 'Training between t = 1 and t = %d \n', pred_par.tmax_training);
        fprintf(fid, 'Evaluation between t = %d and t = %d \n', pred_par.t_eval_start, pred_par.tmax_pred);
        fprintf(fid, 'Horizon of the prediction h = %d \n', pred_par.horizon);
        if pred_par.NORMALIZE_DATA
            fprintf(fid, 'data normalized before prediction\n');
        else
            fprintf(fid, 'data not normalized before prediction \n');
        end
        
        switch(pred_par.pred_meth_idx)
            case 1 %multivariate linear prediction
                fprintf(fid, 'Signal history length k = %d \n', pred_par.SHL);                
            case 2 %RNN with RTRL training
                fprintfRNN_common( fid, pred_par, beh_par )
            case 4 %multivariate linear mean squares (LMS)
                fprintf(fid, 'Signal history length / filter order k = %d \n', pred_par.SHL);  
                fprintf(fid, 'Learning rate / step size eta = %g \n', pred_par.learn_rate);
                fprintfoptim_par( fid, pred_par )              
            case 5 %RNN with UORO training
                fprintfRNN_common( fid, pred_par, beh_par )             
                fprintf(fid, 'Step epsilon used for tangent forward propagation eps_tgt_fwd = %g \n', pred_par.eps_tgt_fwd_prp);
                fprintf(fid, 'Parameter epsilon used when computing the normalizers rho1 and rho2 : eps_nlzer = %d \n', pred_par.eps_normalizers);                
                
        end

end

function fprintfRNN_common( fid, pred_par, beh_par )

    fprintf(fid, 'Signal history length k = %d \n', pred_par.SHL);
    fprintf(fid, 'Nb of neurons in the hidden layer q = %d \n', pred_par.rnn_state_space_dim);
    fprintf(fid, 'Learning rate eta = %g \n', pred_par.learn_rate);
    fprintf(fid, 'Synaptic weights standard deviation (initialization) sg = %g \n', pred_par.Winit_std_dev);
    fprintf(fid, 'Number of runs due to random weights initialization (for computing RMSE of time signals) nb_runs = %d \n', pred_par.nb_runs);
    fprintfoptim_par( fid, pred_par )
    if beh_par.GPU_COMPUTING 
        fprintf(fid, 'Computation with the GPU \n');
    else
        fprintf(fid, 'Computation with the CPU \n');
    end

end

function fprintfoptim_par( fid, pred_par )

    if pred_par.GRAD_CLIPPING % gradient clipping
        fprintf(fid, 'Gradient clipping with threshold grd_tshld = %f \n', pred_par.grad_threshold);
    else
        fprintf(fid, 'No gradient clipping \n');
    end
    switch pred_par.update_meth 
        case 1
            fprintf(fid, 'Optimization : stochastic gradient descent \n');
        case 2
            fprintf(fid, 'Optimization : ADAM, with parameters  \n');
            fprintf(fid, 'beta1 = %f  \n', pred_par.ADAM_beta1);
            fprintf(fid, 'beta2 = %f  \n', pred_par.ADAM_beta2);
            fprintf(fid, 'epsilon = %f  \n', pred_par.ADAM_eps);
    end    

end

