clc, clear
close all
%% Load the hyperspectral image and ground truth
addpath('../functions')
addpath('../data')
load Segundo
load ../coarse/result_coarse
load ../result/reconstruct_result
tic
[w, h, bs] = size(data);
data = hyperNormalize( data );
data_r = hyperConvert2d(data)';
reconstruct_result = hyperNormalize(reconstruct_result);
%% Parameters setup
lamda = 10;
max = 4;
%% Difference
for i = 1: w*h
    sam(i)= hyperSam(data_r(i,:), reconstruct_result(i,:));
end 
sam = reshape(sam , w, h);
SAM = hyperNormalize( sam );
%% Binary the difference 
output  = nonlinear(result_coarse, lamda, max );
B = SAM.* output;
toc
[FPR,TPR,thre] = myPlot3DROC( map, B);
auc = -trapz(FPR,TPR);
fpr = -trapz(FPR,thre);
figure, imagesc(B), axis image, axis off