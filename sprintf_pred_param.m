function [ pred_param_str ] = sprintf_pred_param(pred_par)
% Returns a character string which contains information concerning the prediction parameters for saving and loading temporary variables
% 
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    if pred_par.NORMALIZE_DATA
        nrm_data_str = string('nrlzed data');
    else
        nrm_data_str = string('no nrmztion');
    end

    switch(pred_par.pred_meth_idx)
        case 1 %multivariate linear regression
            pred_param_str = sprintf('k=%d h=%d tmax_train=%d %s', pred_par.SHL, pred_par.horizon, pred_par.tmax_training, nrm_data_str);
                % I write tmax_training here because mult. lin. regression is an offline method so it is more important
        case {2,5} %RNN
            pred_param_str = sprintf('k=%d q=%d eta=%g sg=%g grd_tshld=%g h=%d %s', pred_par.SHL, pred_par.rnn_state_space_dim, ...
                pred_par.learn_rate, pred_par.Winit_std_dev, pred_par.grad_threshold, pred_par.horizon, nrm_data_str);
            % k = nb of time steps for performing one prediction
            % q = nb of neurons in the hidden layer
            % eta = learning rate
            % sg = standard deviation of the gaussian distribution of the initial weights values
            % grd_tshld = clipping value
        case 3 %no prediction
            pred_param_str = sprintf('h=%d %s', pred_par.horizon, nrm_data_str);
        case 4 % prediction with LMS
            pred_param_str = sprintf('k=%d h=%d eta=%g sg=%g %s', pred_par.SHL, pred_par.horizon, pred_par.learn_rate, pred_par.Winit_std_dev, nrm_data_str);    
    end
        
end