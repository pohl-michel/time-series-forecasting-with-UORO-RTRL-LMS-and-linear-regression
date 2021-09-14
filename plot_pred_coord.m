function plot_pred_coord( pred_pts_pos, true_pts_pos, pred_par, path_par, disp_par, pt_idx, dir_idx, run_idx, SAVE_ONLY)
% Plots the predicted dir_idx-th coordinate of the pt_idx-th point
% 
% Author : Pohl Michel
% Date : August 12th, 2020
% Version : v1.0
% License : 3-clause BSD License

    switch dir_idx 
        case 1
            dir_char = 'x';
        case 2
            dir_char = 'y';
        case 3
            dir_char = 'z';
    end

    pred_pts_pos(1:pred_par.SHL, :, :) = true_pts_pos(1:pred_par.SHL, :, :);
        
    % f = figure;
    f = figure('units','normalized','outerposition',[0 0 1 1]); % for saving figures full screen
    tpred_plot_start = 1 + pred_par.tmax_training;
    plot(tpred_plot_start:pred_par.tmax_pred, pred_pts_pos(tpred_plot_start:pred_par.tmax_pred,pt_idx, dir_idx), 'x', 'Color', 'k')
    hold on
    plot(1:pred_par.tmax_pred, true_pts_pos(:,pt_idx, dir_idx), 'Color', 'k')
    title_str = sprintf('Original and predicted %s coordinate of the %d-th point', dir_char, pt_idx);
    title(title_str);
    ylabel(sprintf('%s coordinate', dir_char));
    xlabel('time');
    legend({'predicted coordinates','true coordinates'},'Location','southwest')
    
    filename_suffix = sprintf('%s pred %s %d-th pt %s %d-th run', path_par.time_series_dir, dir_char, pt_idx, ...
                                                                            sprintf_pred_param(pred_par), run_idx);
    fig_filename = sprintf('%s\\%s.fig', path_par.temp_fig_dir, filename_suffix);
    savefig(f, fig_filename);
    png_filename = sprintf('%s\\%s.png', path_par.temp_im_dir, filename_suffix);
    print(png_filename, '-dpng', disp_par.pred_plot_res);
    
    if SAVE_ONLY
       close(f); 
    end

end

