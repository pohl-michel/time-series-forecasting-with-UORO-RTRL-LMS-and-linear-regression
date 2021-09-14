function [ hppars ] = load_hyperpar_cv_info( pred_par )
% load information about the hyperparameters for performing cross correlation
% hppars = hyper-parameters
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    switch(pred_par.pred_meth_idx)
        case 1 % multivariate linear regression
            
            hppars.nb_runs_cv = 1;            
            hppars.nb_runs_eval_test = 1;
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
            
            hppars.other(1).name = 'SHL';
            hppars.other(1).val = [10, 20, 30, 40, 50, 60, 70, 80, 90];             
            hppars.SHL_hyppar_idx = 1; % used for time calculation analysis                
            
        case 2 % RTRL
          
            hppars.nb_runs_cv = 50;
            hppars.nb_runs_eval_test = 300;
            
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];

            hppars.other(1).name = 'SHL';
            hppars.other(1).val = [10, 25, 40, 55];     
            hppars.SHL_hyppar_idx = 1; % used for time calculation analysis            

            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.02, 0.05, 0.1, 0.2];

            hppars.other(3).name = 'Winit_std_dev';
            hppars.other(3).val = [0.01, 0.02, 0.05];            

            hppars.other(4).name = 'rnn_state_space_dim';
            hppars.other(4).val = [10, 25, 40, 55];            
            hppars.state_space_hyppar_idx = 4; % used for time calculation analysis              
            
        case 3 % no prediction

            hppars.nb_runs_cv = 1;
            hppars.nb_runs_eval_test = 1;            
            
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
            
            hppars.other(1).name = 'SHL';
            hppars.other(1).val = [1]; % The lastest acquired value is used as the predicted value        
            hppars.SHL_hyppar_idx = 1; % used for time calculation analysis                
            
        case 4 % multivariate linear mean squares (LMS)
            
            hppars.nb_runs_cv = 1;
            hppars.nb_runs_eval_test = 1;            
               
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
            
            hppars.other(1).name = 'SHL';           
            hppars.other(1).val = [10, 30, 50, 70, 90];  
            hppars.SHL_hyppar_idx = 1; % used for time calculation analysis                
            
            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2];
            
        case 5 % UORO

            hppars.nb_runs_cv = 50;
            hppars.nb_runs_eval_test = 300;
            
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];

            hppars.other(1).name = 'SHL';
            hppars.other(1).val = [10, 30, 50, 70, 90];
            hppars.SHL_hyppar_idx = 1; % used for time calculation analysis            
            
            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.05, 0.1, 0.2];
            
            hppars.other(3).name = 'Winit_std_dev';
            hppars.other(3).val = [0.02, 0.05];            

            hppars.other(4).name = 'rnn_state_space_dim';
            hppars.other(4).val = [10, 30, 50, 70, 90];
            hppars.state_space_hyppar_idx = 4; % used for time calculation analysis    
            
    end

    hppars.nb_additional_params = numel(hppars.other); 
    hppars.nb_hrz_val = length(hppars.horizon_tab); % number of horizon values tested
    for hppar_idx = 1:hppars.nb_additional_params
        hppars.other(hppar_idx).nb_val = length(hppars.other(hppar_idx).val);
    end
    
    nb_calc_temp = 1;
    for hppar_idx = 1:hppars.nb_additional_params
        nb_calc_temp = nb_calc_temp*hppars.other(hppar_idx).nb_val;
    end
    hppars.nb_calc = nb_calc_temp;
    
end

