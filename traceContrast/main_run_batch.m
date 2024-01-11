%% 3D-t图像处理pipeline(图像处理部分 part1)
clearvars; clc; close all; warning off;dbstop if error;

batchtimes = 100;

f = waitbar(0,'Please wait...');
for ii = 1:batchtimes
    waitbar(ii/batchtimes,f,[num2str(ii),filesep,num2str(batchtimes)]);
    cmd = 'conda activate base  && python train_tracecontrast_standard.py SCN SCN --loader SCN --eval';
    system(cmd);
end
close(f)

disp('All finished!')