% Load the original data (CSV files in the "Original data" folder) and converts it into the mat files used by
% prediction.main and hyperparameter_optimization_main
% The mat files created in the "Original data" folder by "convert_csv_to_mat" correspond to the data.mat files 
% in the "1. Input time series sequences" folder.
%
% Author : Pohl Michel
% Date : May 26, 2021
% Version : v1.0
% License : 3-clause BSD License

clc
clear all
close all

SAVE_DATA = true;
DISPLAY_Z_DATA = true;

%accessing datasets
my_folder = 'Original data';
file_route = fullfile(my_folder, '*.csv');
fnames = dir(file_route);
n_files= length(fnames);

marker_idx = 1;
nb_markers = 3;
seq_count = 1;

for sk = 1: n_files
    
    %reading one data file
    disp('------');
    fprintf(1, 'processing file %s\n', fnames(sk).name);
    namenow = fnames(sk).name;
    [data_now,header] = file_to_matrix([my_folder '/' namenow]);
    data_now = data_now(1:end,:); 
    [time_index_max,dim] = size(data_now);
    disp(['data size: ', num2str(time_index_max),' x ',  num2str(dim)]);
    
    if marker_idx == 1
        data_all_markers = zeros(nb_markers*3, time_index_max-1, 'single');
    end
    
    % new data matrix transpose
    temp_data = transpose(data_now(1:(time_index_max-1) ,3:5));
    data_all_markers( marker_idx               , : ) = temp_data(1,:); %x coord
    data_all_markers( marker_idx + nb_markers  , : ) = temp_data(2,:); %y coord
    data_all_markers( marker_idx + nb_markers*2, : ) = temp_data(3,:); %z coord
    
    % plotting the z coordinate of the current marker whose file is open
    if DISPLAY_Z_DATA
        figure
        plot(temp_data(3,:))
        hold on
    end
    
    marker_idx = marker_idx +1;
    if marker_idx == (nb_markers +1)
        marker_idx = 1;
        if SAVE_DATA
            % save data_all_markers
            Folder = cd;
            %Folder = fullfile(Folder, '..');
            save(fullfile(Folder, sprintf('Original data/sequence_%d.mat', seq_count)), 'data_all_markers');
            seq_count = seq_count +1;
        end
    end
    
end


function [M,heading] = file_to_matrix(name)
% function written by I.Zliobaite as part of the code to support the research article
% "Predicting Respiratory Motion for Real-Time Tumour Tracking in Radiotherapy." ArXiv:1508.00749 [Physics], Aug. 2015. arXiv.org, http://arxiv.org/abs/1508.00749.

    M = [];
    fid = fopen(name);
    heading = fgetl(fid);
    
    %a = fgetl(fid);
    
    while ~feof(fid)
	    a = fgetl(fid);
        ii = find(a=='.');
        a(ii)='';
        ii = find(a==',');
        a(ii)='.';
        ii = find(a==';');
        a(ii) = ',';
        M = [M; str2num(a)];
    end
    
    M = M - ones(size(M,1),1)*M(1,:);
    fclose(fid);

end