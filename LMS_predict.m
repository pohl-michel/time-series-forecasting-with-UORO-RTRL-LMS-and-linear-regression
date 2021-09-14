function [Ypred, avg_pred_time, pred_loss_function] = LMS_predict(pred_par, X, Y)
% prediction with least mean squares
%
% Author : Pohl Michel
% Date : November 14th, 2020
% Version : v1.0
% License : 3-clause BSD License

    [p, M] = size(Y);
    [m1, ~] = size(X); 
    m = m1-1; % input dimension without taking into account the bias unit term
    
    % Variables initialization
    
    Ypred = zeros(p, M, 'single');
    pred_loss_function = zeros(M, 1, 'single');
    pred_time_array = zeros(M, 1, 'single');
    W = zeros(p, m+1, 'single');
    
    for t=1:M

        tic
        
        u = X(:,t);
        Ypred(:,t) = W*u;
        e = Y(:,t) - Ypred(:,t);
        
        W_gradient = - kron(e,u.'); 
        W = update_param_optim(W, W_gradient, pred_par, struct(), t);

        pred_time_array(t) = toc;
        pred_loss_function(t) = 0.5*(e.')*e;
        
    end

    avg_pred_time = mean(pred_time_array);

end

