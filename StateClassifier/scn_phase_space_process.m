%%%%%% 3d scn data process pipeline
clearvars; clc; close all; warning off;dbstop if error;
addpath(genpath('./src'))

%% load data
filePath = '../SCNData/Dataset1_SCNProject.mat'; % please insert input filepath here
frameRate = 0.67;
outPath = './data';
mkdir(outPath)

load(filePath);

%% convert Ca2+ time series into phase-space manifolds
[cellNum,timeline] = size(F_set);
trace_zs_set = cell(cellNum,timeline);
xyz = cell(cellNum,timeline);
for tt = 1:timeline
    for ii = 1:10
        dat = F_set{ii,tt};
        trace_zs = zscore(dat);
        x = 1/frameRate*linspace(1,length(trace_zs),length(trace_zs));
        trace_zs_set{ii,tt} = trace_zs;
        mi = mutual(trace_zs);
        [~,mini] = findpeaks(-mi);
        if isempty(mini) == 1
            mini = 8;      %%%% empircally
        end
        dim = 3;
        tau = mini(1);
        y = phasespace(trace_zs,dim,tau);
        xyz{ii,tt} = y;
    end
end

xyz_len = 170;    %%%% empircally
xyz_trim = cellset2trim(xyz,xyz_len);

%% graph dataset
forPred = reshape(xyz_trim,[],1);
pred_num = length(forPred);
%%% node.csv
graph_id = 1:pred_num;
graph_id = repmat(graph_id,xyz_len,1);
graph_id = reshape(graph_id,pred_num*xyz_len,1);
node_id = 1:xyz_len;
node_id = node_id';
node_id = repmat(node_id,pred_num,1);

feat1 = cell2mat(forPred);
feat = cell(length(feat1),1);
f = waitbar(0,'Please wait...');
for ii = 1:length(feat)
    waitbar(ii/length(feat),f,[num2str(ii),filesep,num2str(length(feat))]);
    feat{ii} = formatConvert(feat1(ii,:));
end
close(f)
outfile = 'nodes.csv';

T = table(graph_id,node_id,feat);
writetable(T,fullfile(outPath,outfile));

%%% edges.csv
graph_id = 1:pred_num;
graph_id = repmat(graph_id,xyz_len-1,1);
graph_id = reshape(graph_id,pred_num*(xyz_len-1),1);
src_id = (1:(xyz_len-1))';
src_id = repmat(src_id,pred_num,1);
dst_id = src_id+1;
feat = ones(length(dst_id),1);
T = table(graph_id,src_id,dst_id,feat);
outfile = 'edges.csv';
writetable(T,fullfile(outPath,outfile));

%%% graphs.csv
graph_id = 1:pred_num;
graph_id = graph_id';
clear feat label
feat{1} = '1,0,0,0,0,0';
feat = repmat(feat,length(graph_id),1);
label = zeros(length(feat),1);

T = table(graph_id,feat,label);
outfile = 'graphs.csv';
writetable(T,fullfile(outPath,outfile));

disp('All finished!')