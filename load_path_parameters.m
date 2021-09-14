function [ path_par ] = load_path_parameters()
% This function returns path_par, which contains information concerning the folders and the files to save or open
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    % date and time
    path_par.date_and_time = sprintf('%s %s', datestr(datetime, 'yyyy - mm - dd HH AM MM'), 'min');

    % directory containing the input sequences
    path_par.parent_seq_dir = '1. Input time series sequences';    
    % directory for saving fig files
    path_par.temp_fig_dir = '2. Prediction results (figures)';
    % directory for saving images files
    path_par.temp_im_dir = '3. Prediction results (images)';
    % directory for saving auxiliary RNN variables
    path_par.temp_var_dir = '4. RNN variables (temp)';
    % directory for saving log files
    path_par.txt_file_dir = '5. Log txt files';

    % check if the directories above exist and create them if they do not
    if ~exist(path_par.temp_fig_dir, 'dir')
       mkdir(path_par.temp_fig_dir)
    end
    if ~exist(path_par.temp_im_dir, 'dir')
       mkdir(path_par.temp_im_dir)
    end    
    if ~exist(path_par.temp_var_dir, 'dir')
       mkdir(path_par.temp_var_dir)
    end  
    if ~exist(path_par.txt_file_dir, 'dir')
       mkdir(path_par.txt_file_dir)
    end     
    
    % time series data directories
    path_par.time_series_dir_tab = [
        string('Ext markers seq 1');
        string('Ext markers seq 2');
        string('Ext markers seq 3');
        string('Ext markers seq 4');
        string('Ext markers seq 5');
        string('Ext markers seq 6');
        string('Ext markers seq 7');
        string('Ext markers seq 8');   
        string('Ext markers seq 9');  
        ];
    
    % time series data filenames
    path_par.time_series_data_filename_suffix = 'data.mat';
    % display parameters filename
    path_par.disp_par_filename = 'disp_par.xlsx';
    % prediction parameters filename
    path_par.pred_par_filename_suffix = 'pred_par.xlsx';
    % txt file with parameters and results
    path_par.log_txt_filename = sprintf('log file %s.txt', path_par.date_and_time);
    % directory containing the results 
    path_par.hyperpar_optim_log_filename = sprintf('optim log file %s.txt', path_par.date_and_time);
    
end
