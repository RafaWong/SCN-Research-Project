%%%%%% 3d scn data process pipeline

clear all;
close all;
clc;

color = [1,86,153;
250,192,15;
243,118,74;
95,198,201;
79,89,100]/255;

%% load data
scn_data_path = '..\SCNData\Dataset1_SCNProject.mat'; % please insert input filepath here
dataset_order = '01'; % please insert order of dataset here

load(scn_data_path);

all_num = size(dff_set, 1);

num_time = 200;
half_num_time = num_time / 2;

poi = zeros(all_num, 3);
for i = 1:all_num
    tmp = cell2mat(POI(i,1));
    poi(i,:) = tmp(1:3);
end
POI = poi;

%% standard
trace = zeros(num_time*24, all_num);

for t = 1:24
    count = 0;
    for i = 1:all_num
        dff = cell2mat(dff_set(i,t));
        if ~isempty(dff)
            trace((1:num_time)+num_time*(t-1), i) = dff;
        end
    end
end

save([dataset_order, '_standard.mat'], 'POI', 'trace');

%% time-sample
trace = zeros(half_num_time*24, all_num);

for t = 1:24 
    count = 0;
    for i = 1:all_num
        dff = cell2mat(dff_set(i,t));
        if ~isempty(dff)
            trace((1:half_num_time)+half_num_time*(t-1), i) = dff(1:2:num_time);
        end
    end
end

save([dataset_order, '_time-sample.mat'], 'POI', 'trace');

%% pc-sample
trace = zeros(num_time*24, all_num);

% 1/2 sample
SAMPLING_SET = ceil(0.5*size(POI,1));
srf = struct('X',POI(:,1),'Y',POI(:,2),'Z',POI(:,3));
ifps = fps_euc(srf,SAMPLING_SET);
tmp_pos = POI(ifps,:);
POI = tmp_pos;
ds_trace = zeros(num_time*24, SAMPLING_SET);

for t = 1:24
    count = 0;
    for i = 1:size(ifps, 2)
        ds_dff = cell2mat(dff_set(ifps(i),t));
        if ~isempty(ds_dff)
            ds_trace((1:num_time)+num_time*(t-1), i) = ds_dff;
        end
    end
end

trace = ds_trace;
save([dataset_order, '_pc-sample.mat'], 'POI', 'trace');

