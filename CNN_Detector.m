% 2015/12/6
% 粗定位文本行的分割（及其可视化）
% 文本行的过滤


function img_result1=CNN_Detector(img)
addpath(genpath('../finetune'));
% load first layer features
load models/detectorCentroids_96.mat
% load detector model
load models/CNN-B256.mat
% img = imread('models/sampleImage.jpg');
% fprintf('Constructing filter stack...\n');
filterStack = cstackToFilterStack(params, netconfig, centroids, P, M, [2,2,256]);
img_hight=size(img,1);
% scales=[31.0/img_hight,32.0/img_hight,33.0/img_hight,34.0/img_hight,36.0/img_hight];
scales=32.0/img_hight;
% scales = [1.5,1.2:-0.1:0.1];
% fprintf('Computing responses...\n');
[responses ,scales]= computeResponses(img, filterStack,scales);
% fprintf('Finding lines...\n');
img_result1 = findBoxesFull(responses,scales);
img_result2=visualizeBoxes_12_3(img, img_result1);

% if exist('outputDir')
%   system(['mkdir -p ', outputDir]);
%   save([outputDir, '/output.mat'], 'filterStack', 'responses', 'boxes', '-v7.3');
% end
