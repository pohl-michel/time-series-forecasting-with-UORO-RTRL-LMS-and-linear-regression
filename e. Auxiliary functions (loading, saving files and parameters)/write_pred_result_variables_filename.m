function [str] = write_pred_result_variables_filename(path_par, pred_par)
% returns the filename for the file containing the prediction results 
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    str = sprintf('%s\\pred_result_variables %s %s.mat', path_par.temp_var_dir, path_par.time_series_dir, sprintf_pred_param(pred_par));

end

