function plot_pred_error( RNNloss, disp_par, pred_par, path_par, filename_suffix, title_str, SAVE_ONLY )
% Plots the instantaneous error function as a function of time.
% 
% Author : Pohl Michel
% Date : August 12th, 2020
% Version : v1.0
% License : 3-clause BSD License

    f = figure;
    %f = figure('units','normalized','outerposition',[0 0 1 1]); % for saving figures full screen
    plot(RNNloss, 'Color', 'k')
    
    %t_start_pred = pred_par.t_eval_start;
    %line([t_start_pred, t_start_pred], get(gca, 'ylim'), 'Color', [1 0 0], 'linewidth', disp_par.pred_start_linewidth);
    %yl = ylim();
    %text(t_start_pred, yl(2)*0.9, {'Test data'})
    
    %title(title_str);
    ylabel(sprintf('Loss function (mm^2)'));
    xlabel('Time index');
    
    fig_filename = sprintf('%s\\%s.fig', path_par.temp_fig_dir, filename_suffix);
    savefig(f, fig_filename);
    png_filename = sprintf('%s\\%s.png', path_par.temp_im_dir, filename_suffix);
    print(png_filename, '-dpng', disp_par.pred_plot_res);
    
    if SAVE_ONLY
       close(f); 
    end
    
end
