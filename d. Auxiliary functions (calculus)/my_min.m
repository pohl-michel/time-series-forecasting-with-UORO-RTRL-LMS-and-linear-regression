function [min_A] = my_min(A)
% return the minimum of an array
% This function is already implemented in Matlab 2018 but not in former versions
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    nb_dim = ndims(A);
    min_A = A;
    for k =1:nb_dim
        min_A = min(min_A);
    end

end

