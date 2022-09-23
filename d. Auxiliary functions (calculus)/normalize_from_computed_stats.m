function [ Z ] = normalize_from_computed_stats( X, Mu, Sg)
% X is a data matrix of size (m,p) ie m individuals and p variables
% "normalize_from_computed_stats" returns the standardized matrix Z = (z1, ..., zp) such that 
% Mu(k) = E[zk] = 0 and Sg(k) = sqrt(V(zk)) = 1 for each k in 1,..., p.
% 
% Author : Pohl Michel
% Date : August 12th, 2020
% Version : v1.0
% License : 3-clause BSD License

[~, m] = size(X);

% mean computation
Y = X-Mu*ones(1,m);

% matrix centering
Z = Y./(Sg*ones(1,m));