function [ disp_par ] = load_display_parameters(path_par)
% Load the parameters concerning display,
% which are initially stored in the file named path_par.disp_par_filename.
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    opts = detectImportOptions(path_par.disp_par_filename);
    opts = setvartype(opts,'double');
    opts.DataRange = '2:2';
    disp_par = table2struct(readtable(path_par.disp_par_filename, opts));
    disp_par.pred_plot_res = sprintf('-r%d', int16(disp_par.pred_plot_res));


end