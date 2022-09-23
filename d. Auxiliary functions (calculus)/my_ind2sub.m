function [ tab_idx ] = my_ind2sub( size_tab, lin_idx )
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    N = numel(size_tab);
    tab_idx = zeros(N, 1);
    
    inputs = repmat({1},1, N);
    for dim_idx = 1:N
        inputs{dim_idx} = 1:size_tab(dim_idx);
    end

    outputs = cell(1, N);
    [outputs{:}] = ndgrid(inputs{:});
    
    for k = 1:N
        tab_idx(k) = outputs{k}(lin_idx);
    end    
    
end

