function [pred_par] = load_pred_par(path_par)
% Load the parameters concerning prediction,
% which are initially stored in the file named path_par.pred_par_filename.
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    path_par.pred_par_filename = sprintf('%s\\%s', path_par.input_seq_dir, path_par.pred_par_filename_suffix);

    opts = detectImportOptions(path_par.pred_par_filename);
    opts = setvartype(opts,'double');
    opts.DataRange = '2:2';
    pred_par = table2struct(readtable(path_par.pred_par_filename, opts));
    
    % Choice of the prediction method
    pred_par.pred_meth_idx = 5; 
    
    switch(pred_par.pred_meth_idx)
        case 1
            pred_par.pred_meth_str = 'multivar lin reg';
            pred_par.nb_runs = 1; % because it is not a stochastic method
            pred_par.NORMALIZE_DATA = false;
            pred_par.tmax_training = 540;
        case 2
            pred_par.pred_meth_str = 'RTRL_RNN';
            pred_par.NORMALIZE_DATA = true;
            pred_par.update_meth = 1; % gradient descent
            pred_par.GRAD_CLIPPING = true;
            pred_par.grad_threshold = 2.0; 
        case 3
            pred_par.pred_meth_str = 'no prediction';
            pred_par.nb_runs = 1;
            pred_par.NORMALIZE_DATA = false;
            pred_par.SHL = 1; % The lastest acquired value is used instead of the predicted value         
        case 4
            pred_par.pred_meth_str = 'LMS'; %multivariate least mean squares
            pred_par.nb_runs = 1; % not a stochastic method
            pred_par.NORMALIZE_DATA = true;    
            pred_par.update_meth = 1; % gradient descent
            pred_par.GRAD_CLIPPING = true;
            pred_par.grad_threshold = 2.0;  
        case 5
            pred_par.pred_meth_str = 'UORO_RNN';
            pred_par.NORMALIZE_DATA = true;
            pred_par.eps_tgt_fwd_prp = 0.0000001;
            pred_par.eps_normalizers = 0.0000001;
            pred_par.update_meth = 1;
            pred_par.GRAD_CLIPPING = true;
            pred_par.grad_threshold = 2.0;           
    end
    
    if isfield(pred_par, 'update_meth')
        switch(pred_par.update_meth)
            case 1
                pred_par.update_meth_str = 'stochastic gradient descent';
            case 2
                pred_par.update_meth_str = 'ADAM (adaptive moment estimation)';
                pred_par.ADAM_beta1 = 0.9;
                pred_par.ADAM_beta2 = 0.999;
                pred_par.ADAM_eps = 10^-8;
        end    
    end
    
end