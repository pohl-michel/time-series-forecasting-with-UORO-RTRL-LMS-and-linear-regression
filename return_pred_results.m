function [pred_pts_pos, avg_pred_time_run_idx, pred_loss_functionv2 ] = return_pred_results( pred_par, run_idx, Ypred, avg_pred_time, pred_loss_function )
% reshapes/reformats the prediction results from the variables Ypred and pred_loss_function 
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    Ytemp = transpose(Ypred(:,:,run_idx));
    [M,p] = size(Ytemp);
    nb_pts = p/3;
    tmax = pred_par.tmax_pred;

    pred_pts_pos = zeros(tmax, nb_pts, 3); 
    pred_pts_pos((tmax-M+1):tmax,:,1) = Ytemp(:,1:nb_pts);                %x coordinates of the points 
    pred_pts_pos((tmax-M+1):tmax,:,2) = Ytemp(:,(nb_pts+1):(2*nb_pts));   %y coordinates of the points 
    pred_pts_pos((tmax-M+1):tmax,:,3) = Ytemp(:,(2*nb_pts+1):(3*nb_pts)); %z coordinates of the points    

    pred_loss_functionv2 = zeros(tmax, 1, 'single');
    pred_loss_functionv2((tmax-M+1):tmax) = pred_loss_function(:, run_idx);
 
    avg_pred_time_run_idx = avg_pred_time(run_idx);
    
end