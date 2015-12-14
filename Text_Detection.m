clc
clear
warning off all
File_Path ='E:\2015 text detection£¨EdgeBox£©';
addpath(genpath([File_Path,'\Related resources\piotr_toolbox']));

%% Parameters for EdgeBox 
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .02;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect

%%

addpath([File_Path,'\Addational matlab toolbox\DeepLearnToolbox-master\CNN']);
addpath([File_Path,'\Addational matlab toolbox\DeepLearnToolbox-master\util']);
CNNModel=load([File_Path,'\Trained Models for Classification\CNN_64_128_100.mat']);
for file_num =2:233
    Image_Path =[File_Path,'\test images\ICDAR 2013 ²âÊÔ¼¯\Challenge2_Test_Task12_Images\img_',num2str(file_num),'.jpg'];
    g = imread(Image_Path);
    I = double(rgb2gray(g));
    [len,wid,~] = size(g);
    tic, bbs=edgeBoxes(g,model,opts); 
    bbs(:,3) = bbs(:,1) + bbs(:,3)-1; %% x y w h ---> x1 y1 x2 y2
    bbs(:,4) = bbs(:,2) + bbs(:,4)-1;
    num_bb = size(bbs,1);
    Data = zeros(32*32,num_bb);
    for ii = 1:num_bb
        temp = I(bbs(ii,2):bbs(ii,4),bbs(ii,1):bbs(ii,3));
        temp = imresize(temp,[32 32]);
        Data(:,ii) = temp(:);
    end
    Data = Data/255;
    temp = mean(Data);
    Data = Data - repmat(temp,1024,1);
    Data = reshape(Data,[32  32 num_bb]);
    
    Prob = zeros(1,num_bb);
    it_num = ceil(num_bb/5000);
    for ii = 1:it_num
        x1 = (ii-1)*5000+1;
        x2 = min(ii*5000,num_bb);
        temp = Data(:,:,x1:x2);
        temp = cnnff2(CNNModel.cnn,temp);
        Prob(x1:x2) = temp.o(1,:);
    end
    toc
  
    Out_Image = zeros(len,wid);
    for ii = 1:length(Prob) 
       Out_Image(bbs(ii,2):bbs(ii,4),bbs(ii,1):bbs(ii,3)) = max(Prob(ii),Out_Image(bbs(ii,2):bbs(ii,4),bbs(ii,1):bbs(ii,3)));
    end
   imshow(Out_Image)
end