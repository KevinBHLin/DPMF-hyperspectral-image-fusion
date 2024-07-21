clear all
close all
clc
% addpath(genpath('CNN_subspace'))
%--------------------------------------------------------------
% Simulated Nikon D700 spectral response T = [R;G;B], 
% Please note, the same transformation must be used between 'the ground
% truth and RGB image' and the leanred and the transformed dictionary.
%--------------------------------------------------------------
T1 = [0.005 0.007 0.012 0.015 0.023 0.025 0.030 0.026 0.024 0.019 0.010 0.004   0     0      0    0     0     0     0     0     0     0     0     0     0     0     0     0    0     0       0];
T2 = [0.000 0.000 0.000 0.000 0.000 0.001 0.002 0.003 0.005 0.007 0.012 0.013 0.015 0.016 0.017 0.02 0.013 0.011 0.009 0.005  0.001  0.001  0.001 0.001 0.001 0.001 0.001 0.001 0.002 0.002 0.003 ];
T3 = [0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.001 0.003 0.010 0.012  0.013  0.022  0.020 0.020 0.018 0.017 0.016 0.016 0.014 0.014 0.013];
F  = [T1;T2;T3];


img_path = '.\data\cave';
FileList = dir(img_path);
folderlength = 0;

for i=1:length(FileList)
    if ( FileList(i).isdir==1 && ~strcmp(FileList(i).name,'.') && ~strcmp(FileList(i).name,'..') )
        folderlength = folderlength + 1;
        FileFolder{folderlength} = [FileList(i).name];
    end
end

RMSE = zeros(31,2);
SAM = zeros(31,2);
noise_max_list = cell(32,1);
sigma_max_list = cell(32,1);
noise_min_list = cell(32,1);
sigma_min_list = cell(32,1);
Indexes = zeros(32,8);
point = 1;
Time = zeros(32,2);
result = cell(32,1);
for t = [1]
%--------------------------------------------------------------
% Read the input image
%--------------------------------------------------------------    
    for j=1:31
        str=strcat(img_path, '\', FileFolder{t}, '\', FileFolder{t}, '\',...
                FileFolder{t}, '_', num2str(floor(j/10)), num2str(mod(j,10)), '.png');
            ddd = imread(str);
        I_REF(:,:,j) = ddd(:,:,1);
    end
    
% im=uint16(im(15+1:15+320,15+1:15+320,11:end));  %remove the first 11 bands 

% X=X+mean(X(:)); % This is adding an offset to make sure there is no negative data
% X=X/(max(X(:))+min(X(:)));

% enviwrite(im,info,'oriim');
I_REF = im2double(I_REF);
% if t==32
%     I_REF = double(I_REF)./255;
% else
%     I_REF = double(I_REF)./65535;%max(double(I_REF(:)));%im2double(I_REF);
% end
% problem setting (please change depending on your problem setting)


%ratio = 8; % difference of GSD between MS and HS
%size_kernel=[16 16];

% ratio = 4; % difference of GSD between MS and HS
% size_kernel=[8 8];

ratio = 2; % difference of GSD between MS and HS
size_kernel=[2*ratio-1 2*ratio-1];

sig = (1/(2*(2.7725887)/ratio^2))^0.5;
start_pos(1)=1; % The starting point of downsampling
start_pos(2)=1; % The starting point of downsampling
% [I_HS,KerBlu]=conv_downsample(I_REF,ratio,size_kernel,sig,start_pos);


KerBlu = fspecial('Gaussian',[size_kernel(1) size_kernel(2)],sig);
I_HS=imfilter(I_REF, KerBlu, 'circular');
I_HS=I_HS(start_pos(1):ratio:end, start_pos(2):ratio:end,:);

% I_HS = imresize(I_REF,1./ratio);
I_MS = reshape((F*reshape(I_REF,[],size(I_REF,3))')',size(I_REF,1),size(I_REF,2),[]);

snrMSdB = 40; snrHSdB = 35;
%add noise
I_HSn = I_HS; I_MSn = I_MS;
sigma_hsi = zeros(size(I_HSn,3),1);
sigma_msi = zeros(size(I_MSn,3),1);
snrHS = 10^(snrHSdB/20);
snrMS = 10^(snrMSdB/20);
for i = 1:size(I_HSn,3)
    mu = mean(reshape(I_HSn(:,:,i),1,[]));
    sigma_hsi(i) = mu/snrHS;
    I_HSn(:,:,i) = I_HSn(:,:,i) + randn(size(I_HSn,1),size(I_HSn,2))*sigma_hsi(i);
end
for i = 1:size(I_MSn,3)
    mu = mean(reshape(I_MSn(:,:,i),1,[]));
    sigma_msi(i) = mu/snrMS;
    I_MSn(:,:,i) = I_MSn(:,:,i) + randn(size(I_MSn,1),size(I_MSn,2))*sigma_msi(i);
end

Truth = hyperConvert2d(I_REF);
[M,N,L] = size(I_REF);



%% NMF-DPR
point = point + 1;
t0=clock;
[E,A] = NMF_DPR(I_HSn, I_MSn, I_REF, F, KerBlu, start_pos, sigma_hsi, sigma_msi, snrMSdB, snrHSdB);
Z6 = hyperConvert3d(E*A);
Time(point,1)=etime(clock,t0);
[psnr6,rmse6, ergas6, sam6, uiqi6,ssim6,DD6,CC6] = quality_assessment(I_REF.*255, Z6.*255, 0, 1.0/ratio);
% Perform the evaluation
Indexes(point,1) = psnr6;
Indexes(point,2) = rmse6;
Indexes(point,3) = ergas6;
Indexes(point,4) = sam6;
Indexes(point,5) = uiqi6;
Indexes(point,6) = ssim6;
Indexes(point,7) = DD6;
Indexes(point,8) = CC6;
result{point} = Z6;

%% DAPMF+FFDNet
point = point + 1;
t0=clock;
lambda_u = 10^1;
lambda_v = 7;
d = 16;
r = 25;
iter_num = 15;
inner_iter_num = 2;
[E,A, noise, sigma] =  PMF_FFDNet_Fusion0809(I_HSn, I_MSn, I_REF, F, KerBlu, start_pos, sigma_hsi, sigma_msi, lambda_u, lambda_v, r, d,iter_num, inner_iter_num);
Z6 = hyperConvert3d(E*A);
Time(point,1)=etime(clock,t0);
[psnr6,rmse6, ergas6, sam6, uiqi6,ssim6,DD6,CC6] = quality_assessment(I_REF.*255, Z6.*255, 0, 1.0/ratio);
% Perform the evaluation
Indexes(point,1) = psnr6;
Indexes(point,2) = rmse6;
Indexes(point,3) = ergas6;
Indexes(point,4) = sam6;
Indexes(point,5) = uiqi6;
Indexes(point,6) = ssim6;
Indexes(point,7) = DD6;
Indexes(point,8) = CC6;
result{point} = Z6;

%% DAPMF+DnCNN
point = point + 1;
t0=clock;
lambda_u = 10^1;
lambda_v = 1;
d = 16;
r = 25;
iter_num = 15;
inner_iter_num = 2;
[E,A, noise, sigma] =  PMF_DnCNN_Fusion3(I_HSn, I_MSn, I_REF, F, KerBlu, start_pos, sigma_hsi, sigma_msi, lambda_u, lambda_v, r, d, iter_num, inner_iter_num);
Z6 = hyperConvert3d(E*A);
Time(point,1)=etime(clock,t0);
[psnr6,rmse6, ergas6, sam6, uiqi6,ssim6,DD6,CC6] = quality_assessment(I_REF.*255, Z6.*255, 0, 1.0/ratio);
% Perform the evaluation
Indexes(point,1) = psnr6;
Indexes(point,2) = rmse6;
Indexes(point,3) = ergas6;
Indexes(point,4) = sam6;
Indexes(point,5) = uiqi6;
Indexes(point,6) = ssim6;
Indexes(point,7) = DD6;
Indexes(point,8) = CC6;
result{point} = Z6;


%% DAPMF+DnCNN*
point = point + 1;
t0=clock;
lambda_u = 10^1;
lambda_v = 1;
d = 16;
r = 25;
iter_num = 15;
inner_iter_num = 2;
modelfilepath = 'NetworkModel\\NetworkTrainedHyperspectralImage\\result.mat'
[E,A, noise, sigma] =  PMF_DnCNN_Fusion3(I_HSn, I_MSn, I_REF, F, KerBlu, start_pos, sigma_hsi, sigma_msi, lambda_u, lambda_v, r, d, iter_num, inner_iter_num, modelfilepath);
Z6 = hyperConvert3d(E*A);
Time(point,1)=etime(clock,t0);
[psnr6,rmse6, ergas6, sam6, uiqi6,ssim6,DD6,CC6] = quality_assessment(I_REF.*255, Z6.*255, 0, 1.0/ratio);
% Perform the evaluation
Indexes(point,1) = psnr6;
Indexes(point,2) = rmse6;
Indexes(point,3) = ergas6;
Indexes(point,4) = sam6;
Indexes(point,5) = uiqi6;
Indexes(point,6) = ssim6;
Indexes(point,7) = DD6;
Indexes(point,8) = CC6;
result{point} = Z6;

end


