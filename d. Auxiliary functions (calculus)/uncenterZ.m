function [ X ] = uncenterZ( Z, Mu, Sg )
% Z is a data matrix with standardized variables 
% ie Z = [z1, ..., zp] where
% Mu(k) = E[zk] = 0 and Sg(k) = sqrt(V(zk)) = 1 for each k in 1,..., p.
% Returns the uncentered data matrix using the means and standard deviations vectors Mu and Sg 
% (cf function standard_scores)
% 
% Author : Pohl Michel
% Date : August 12th, 2020
% Version : v1.0
% License : 3-clause BSD License

[~, m] = size(Z);
X = Mu*ones(1,m,'single') + (Sg*ones(1,m,'single')).*Z;


end

