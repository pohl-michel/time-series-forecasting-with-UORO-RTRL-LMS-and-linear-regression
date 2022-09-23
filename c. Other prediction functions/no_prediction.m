function [Ypred, avg_pred_time, pred_loss_function] = no_prediction(pred_par, X, Y)
% case when we do not perform prediction
% 
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

avg_pred_time = 0;

k = pred_par.SHL;
[data_dim, ~] = size(Y);

Ypred = X((2+(k-1)*data_dim):(1+k*data_dim),:);
pred_loss_function = transpose(sum((Ypred - Y).^2, 1));


end

