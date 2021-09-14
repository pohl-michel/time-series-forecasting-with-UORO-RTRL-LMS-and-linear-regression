function [ Z, Mu, Sg ] = standard_scores( X )
% X is a data matrix of size (p,m) ie m individuals and p variables
% standard_scores returns the standardized matrix Z = (z1, ..., zp) such that 
% Mu(k) = E[zk] = 0 and Sg(k) = sqrt(V(zk)) = 1 for each k in 1,..., p.
% 
% Author : Pohl Michel
% Date : August 12th, 2020
% Version : v1.0
% License : 3-clause BSD License

[~, m] = size(X);

% mean computation
Mu = mean(X,2);
Y = X-Mu*ones(1,m, 'single');

% standard deviations
Sg = sqrt(mean(Y.^2, 2));

% matrix centering
Z = Y./(Sg*ones(1,m, 'single'));

end

