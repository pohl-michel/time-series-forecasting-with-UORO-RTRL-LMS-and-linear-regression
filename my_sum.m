function [my_sum] = my_sum(A)
% return the sum of the elements of an array
% This function is already implemented in Matlab 2018 but not in former versions
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    nb_dim = ndims(A);
    my_sum = A;
    for k =1:nb_dim
        my_sum = sum(my_sum);
    end

end

