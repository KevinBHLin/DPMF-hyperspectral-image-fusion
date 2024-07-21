
function [U,V, noise_list, sigma_list] = PMF_FFDNet_Fusion0809(HSI, MSI, Truth, F, psf, start_pos, sigma_hsi, sigma_msi, lambda_u, lambda_v, r, d, iter_num, inner_iter_num, modelfilepath)

% [FusionImage, BlurVersion, ResOutline, ResOutFrame, val_RMSE, RMSE_ITER, ConvegenIter, H, U_avr, W_avr, V_avr] = BSDMF_FAST( HSI, MSI, F, 0, 'SVD', 10, 30, 15, 20, Truth, 0);

if(nargin<15)
    modelfilepath = '.\FFDNet-master\FFDNet-master\models\FFDNet_gray.mat';  
end

maxIter = iter_num;
useGPU = 1;
epsilon = 0.0001;
rank1 = d;
rank2 = r;
resAB(1) = 1;

[m,n,L] = size(HSI);
[M,N,l] = size(MSI);
size_hsi = [m,n];
size_msi = [M,N];

%     load('NetworkModel\\NetworkTrainedGrayImage\\result.mat');
%     load('NetworkModel\\NetworkTrainedHyperspectralImage\\result.mat');
    global sigmas
    load(modelfilepath,'net')
%     disp(['The CNN is trained by Gray Images'])

    if useGPU
        net = vl_simplenn_move(net, 'gpu') ;
    end


% [U,W] = LSMMUnmixing(HSI, sigma_hsi, rank, imresize(Truth,[m,n]),1);
% V = hyperConvert2d(imresize(hyperConvert3d(W,m),[M,N]));
    


Y1 = hyperConvert2d(HSI);
Y2 = hyperConvert2d(MSI);
Truth = hyperConvert2d(Truth);

if rank1>=rank2
    [H,~,~] = svds(Y1, rank1);
    U = eye([rank1,rank2]);
    W = H(:,1:rank2)'*Y1;
    V = H(:,1:rank2)'*hyperConvert2d(imresize(HSI,[M,N]));
else
    [H,~,~] = svds(Y1, rank2);
    U = eye([rank1,rank2]);
    W = H'*Y1;
    V = H'*hyperConvert2d(imresize(HSI,[M,N]));
    H = H(:,1:rank1);
end

% U = sisal(Truth, rank, 'spherize', 'no','MM_ITERS',80, 'TAU', 0.006, 'verbose',0);
% % [U, indicies] = vca(Truth, rank);
% % U(U<0) = 0; U(U>1) = 1;
% V = sunsal(U, Truth,'POSITIVITY','yes','ADDONE','yes');
% W = hyperConvert2d(imresize(hyperConvert3d(V,M),[m,n]));



% Initialisations
% U = sisal(Y1, rank, 'spherize', 'no','MM_ITERS',80, 'TAU', 0.006, 'verbose',0);
% W = sunsal(U, Y1,'POSITIVITY','yes','ADDONE','yes');
% stdW = std(W');
% W = bsxfun(@rdivide,W,stdW');
% U = bsxfun(@times,U,stdW);
% V = hyperConvert2d(imresize(hyperConvert3d(W,m),[M,N]));


Y1 = bsxfun(@rdivide,Y1,sigma_hsi); 
Y2 = bsxfun(@rdivide,Y2,sigma_msi); 

H = bsxfun(@rdivide,H,sigma_hsi);
F = bsxfun(@rdivide,F,sigma_msi);
F = bsxfun(@times,F',sigma_hsi)';

resAB(1) =  norm(Y1-H*U*W,'fro')^2+norm(Y2-F*H*U*V,'fro')^2;

RMSE(1) = hyperErrRMSE(Truth, bsxfun(@times,H*U*V,sigma_hsi));
fprintf(['Iter: ' num2str(0) ' RMSE: ' num2str(RMSE(1)) '\n'])
Uth = sigma_hsi;

F_ori = F; U_ori = U;

%     W = W.*sqrt(norm( U_ori*U_ori' ,'fro'));
%     V = V.*sqrt(norm( F*U_ori*U_ori'*F' ,'fro'));
%     U = U./sqrt(norm( U_ori*U_ori' ,'fro'));
%     F = F.*sqrt(norm( U_ori*U_ori' ,'fro'))./sqrt(norm( F_ori*U_ori*U_ori'*F_ori' ,'fro'));
%     
%     resAB(1) =  norm(Y1-U*W,'fro')^2+norm(Y2-F*U*V,'fro')^2;
%     RMSE(1) = hyperErrRMSE(Truth, bsxfun(@times,U*(V./sqrt(norm( F_ori*U_ori*U_ori'*F_ori' ,'fro')).*sqrt(norm( U_ori*U_ori' ,'fro'))),sigma_hsi));
%     fprintf(['Iter: ' num2str(0) ' RMSE: ' num2str(RMSE(1)) '\n'])

resAB(1) =  norm(Y1-H*U*W,'fro')^2+norm(Y2-F*H*U*V,'fro')^2;
RMSE(1) = hyperErrRMSE(Truth, bsxfun(@times,H*U*V,sigma_hsi));
fprintf(['Iter: ' num2str(0) ' RMSE: ' num2str(RMSE(1)) '\n'])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

U_old = U;
W_old = W;
V_old = V;

par.psf = psf;
par.fft_B      =    psf2otf(psf,[M N]);
par.fft_BT     =    conj(par.fft_B);

SS = zeros(size_msi(1),size_msi(2),size(U,2));
SS(start_pos(1):size_msi(1)/size_hsi(1):end, start_pos(2):size_msi(2)/size_hsi(2):end, :) = 1;
YSGT = zeros(size_msi(1),size_msi(2),size(Y1,1));
YSGT(start_pos(1):size_msi(1)/size_hsi(1):end, start_pos(2):size_msi(2)/size_hsi(2):end, :) = permute(reshape(Y1,[size(Y1,1),size_hsi(1),size_hsi(2)]),[2,3,1]);
YSGT = real(ifft2(par.fft_BT.*fft2(YSGT)));
YSGT = reshape(permute(YSGT,[3,1,2]),[size(Y1,1),size(Y2,2)]);
par.SS = SS;
par.YSGT = YSGT;


par.start_pos = start_pos;
par.sigma_hsi = sigma_hsi;
par.sigma_msi = sigma_msi;
par.size_hsi = size_hsi;
par.size_msi = size_msi;
par.Uth = Uth;

par.inner_iter_num = inner_iter_num;

par.lambda_u = lambda_u;%15;%1e1;
par.lambda_v = lambda_v;

par.Fth = ones(size(F));
par.Fth = bsxfun(@rdivide,par.Fth,sigma_msi);
par.Fth = bsxfun(@times,par.Fth',sigma_hsi)';
par.H = H;
F_old = F;

noise_list = zeros(maxIter,1);
sigma_list = zeros(maxIter,1);

for j=1:maxIter
%     W = W.*sqrt(norm( U*U' ,'fro'));
%     V = V.*sqrt(norm( F*U*U'*F' ,'fro'));
%     U = U./sqrt(norm( U*U' ,'fro'));
%     F = F.*sqrt(norm( U*U' ,'fro'))./sqrt(norm( F*U*U'*F' ,'fro'));

    
    % update the abundance matrix W and V
%     if j<300
%         [~, W, V, res, W_old, V_old] = UpdateWandVStep2(Y1, Y2, U, V, W, F, Blur, start_pos, sigma_hsi, sigma_msi, net, size_hsi, size_msi, useGPU, j, W_old, V_old);
%     else
        [~, W, V, res, W_old, V_old, noise, sigma] = UpdateWandVStep(Y1, Y2, H*U, V, W, F, par, net, useGPU, j, W_old, V_old);
%     end
    noise_list(j) = noise;
    sigma_list(j) = sigma;
    resB2(j) = min(res);
    
    resAB(j+1) =  norm(Y1-H*U*W,'fro')^2+norm(Y2-F*H*U*V,'fro')^2; %resA2(j)+resB2(j);
    
    % update the spectral matrix U
    [U, res, U_old] = UpdateUStep(Y1, Y2, U, V, W, F, par, j, U_old);
    resA2(j) = min(res);
    
    % Residual of the objective function (5a)
    resAB(j+1) =  norm(Y1-H*U*W,'fro')^2+norm(Y2-F*H*U*V,'fro')^2 + par.lambda_u*norm(U,'fro')^2; %resA2(j)+resB2(j);
    
%     [F, res, F_old] = UpdateFStep( Y1, Y2, U, V, W, F, par, j, F_old );
%     
%     resAB(j+1) =  norm(Y1-U*W,'fro')^2+norm(Y2-F*U*V,'fro')^2 + par.lambda_u*norm(bsxfun(@times,U,sigma_hsi),'fro')^2; %resA2(j)+resB2(j);
    
    % Compute RMSE only for printing during procedure
    RMSE(j) = hyperErrRMSE(Truth, bsxfun(@times,H*U*V,sigma_hsi));
%     RMSE(j) = hyperErrRMSE(Truth, bsxfun(@times,U*(V./sqrt(norm( F_ori*U_ori*U_ori'*F_ori' ,'fro')).*sqrt(norm( U_ori*U_ori' ,'fro'))),sigma_hsi));
    
    % Convergence checks
    if ( resAB(j) / resAB(j+1) ) > 1+epsilon || ( resAB(j) / resAB(j+1) ) < 1-epsilon
        fprintf(['Iter: ' num2str(j) ' RMSE: ' num2str(RMSE(j)) '\n'])
    else
        fprintf(['Iter: ' num2str(j) ' RMSE: ' num2str(RMSE(j)) '\n'])
        fprintf(['Stopped after ' num2str(j) ' iterations. Final RMSE: ' num2str(RMSE(j)) '\n'])
        break
    end
end
% U = bsxfun(@times,U,sigma_hsi).*sqrt(norm( U_ori*U_ori' ,'fro'));
% V = V./sqrt(norm( F_ori*U_ori*U_ori'*F_ori' ,'fro'));

U = bsxfun(@times,H*U,sigma_hsi);
noise_list = noise_list(1:j,1);
sigma_list = sigma_list(1:j,1);

end


function [ U, res, U_old] = UpdateUStep( Y1, Y2, U, V, W, F, par, Iter_count, U_old )
% Solving eq. (6) with a projected gradient descent method.

[L,r] = size(U);
A = kron(W*W',par.H'*par.H) + kron(V*V',par.H'*F'*F*par.H) + kron(eye(r), par.lambda_u*eye(L));
B = reshape(par.H'*Y1*W' + par.H'*F'*Y2*V',[L*r,1]);
U = reshape(A\B,[L,r]);
U_old = U;

res = norm(Y1-par.H*U*W,'fro')^2+norm(Y2-F*par.H*U*V,'fro')^2;

end

function [ F, res, F_old] = UpdateFStep( Y1, Y2, U, V, W, F, par, Iter_count, F_old )
% Solving eq. (6) with a projected gradient descent method.

maxIter = 1;
epsilon = 1.01;
gamma1 = 1.01;

notfirst = 0;

res(1) = sqrt(norm(Y2-F*U*V,'fro')^2/size(Y2,1)/size(Y2,2))+100;

for k=1:maxIter
    F1 = F;
    
    % 2.1. Update of signatures
%     ck = gamma1 * norm(W*W','fro') + gamma1 * norm(V*V','fro');
%     E = U - 1/ck * ( U*W - Y1 ) * W' - 1/ck * F' * ( F*U*V - Y2 ) * V';
%     ck = gamma1 * norm(W*W'+lambda*DD,'fro');
    ck = gamma1 * norm(U*V*V'*U','fro');
%     F = (Iter_count-1)/(Iter_count+2)*(F-F_old) + F;
    F_old = F1;
%     E = U - 1/ck * ( U*W - Y1 ) * W' - 1/ck * U*DD;
    E = F - 1/ck * ( F*U*V - Y2 ) * V'*U';
    F = PplusaF(E, par.th);
    
    % Calculation of residuals
    res(k+1,1) = sqrt(norm(Y2-F*U*V,'fro')^2/size(Y2,1)/size(Y2,2));
    
    % Checks for exiting iteration
    if (1/res(k+1) * res(k)) < epsilon
        fprintf(['Hyper: Iter ' num2str(k) ', res: ' num2str(res(k+1)*1000) '. '])
        break
    end
    
    if (res(k+1) / res(k))>1
        if notfirst == 1
            F = F_old;
            text = ['Hyper: Exited during ' num2str(k) 'th iteration with residual ' num2str(res(k+1)*1000) ', because res increased'];
            disp(text)
            break
        else
            notfirst = 1;
        end
    end
    
end
end

function F = PplusaF(F, th)
% max{0,U}
F(F<0) = 0;
index = (F-th)>0;
F(index) = th(index);
end


function U = Pplusa(U, th)
% max{0,U}
U(U<0) = 0;
U = bsxfun(@times,U,th);
U(U>1) = 1;
U = bsxfun(@rdivide,U,th);
end

function [ U, W, V, res, W_old, V_old, SNR, sigma ] = UpdateWandVStep( Y1, Y2, U, V, W, F, par, net, useGPU, Iter_count, W_old, V_old )
% Solving eq. (7) with a projected gradient descent method.

maxIter = par.inner_iter_num;
epsilon = 1.01;
gamma2 = 1.01;
%eta = 1; 2022-07-13
eta = par.lambda_v^2;

ek = gamma2 * norm( F*U*U'*F' ,'fro');
% tmp1= (F*U*U'*F');
% tmp2 = (U*U');
% ek = (max(tmp1(:)) + max(tmp2(:)))*gamma2;
noise_p_sigma = sqrt(1/ek)*eta;
sigma = noise_p_sigma;
SNR = 1e10;

global sigmas

res(1) = norm(Y1-U*W,'fro')^2+norm(Y2-F*U*V,'fro')^2+100;


for k=1:maxIter
    V1 = V;
    W1 = W;
    
    % 2.2. Update the AbundancesD
    
    V = (Iter_count-1)/(Iter_count+2)*(V-V_old) + V;
    V_old = V1;
    W_old = W1;
    P = V - 1/ek * U' * F' * ( F*U*V - Y2 );
    
    tmp = permute(reshape(V,[size(V,1),par.size_msi(1),par.size_msi(2)]),[2,3,1]);
    tmp = real(ifft2(par.fft_B.*fft2(tmp)));
    tmp = tmp.*par.SS;
    tmp = real(ifft2(par.fft_BT.*fft2(tmp)));
    tmp = reshape(permute(tmp,[3,1,2]),[size(U,2),par.size_msi(1)*par.size_msi(2)]);
    tmp = U'*(U*tmp-par.YSGT);
    P = P - 1/ek * tmp;
    
    P = reshape(P,[size(P,1),par.size_msi(1),par.size_msi(2)]);
    V = P;
    for jj=1:size(P,1)
        eigen_im=squeeze(  P(jj,:,:));
        min_x = min(eigen_im(:));
        max_x = max(eigen_im(:));
        eigen_im = eigen_im - min_x;
        scale = max_x-min_x;
        eigen_im =single (eigen_im/scale);
        input = gpuArray(eigen_im);
        sigmas = noise_p_sigma^2/scale;
        result    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
        BB = gather(result(end).x);
        V(jj,:,:)=double(BB)*scale + min_x;
        SNR = min(10*log10(mean(BB(:).^2)/noise_p_sigma^2),SNR);
    end
%     V(V<0) = 0; V(V>1) = 1;
    V = permute(V,[2,3,1]);
    W = imfilter(V, par.psf, 'circular');
    W = W(par.start_pos(1):(par.size_msi(1)/par.size_hsi(1)):end, par.start_pos(2):(par.size_msi(2)/par.size_hsi(2)):end,:);
    W = permute(W,[3,1,2]);
    V = permute(V,[3,1,2]);
    W = reshape(W,[size(W,1),par.size_hsi(1)*par.size_hsi(2)]);
    V = reshape(V,[size(V,1),par.size_msi(1)*par.size_msi(2)]);
    
    % Calculation of residuals
    res(k+1,1) = norm(Y1-U*W,'fro')^2+norm(Y2-F*U*V,'fro')^2;
    
    % Checks for exiting iteration
    if (1/ (k+1) * res(k)) < epsilon
        fprintf(['Multi: Iter ' num2str(k) ', res: ' num2str(res(k+1)*1000) '. '])
        break
    end
    
    if (res(k+1) / res(k))>1
        W = W_old;
        V = V_old;
        text = ['Multi: Exited during ' num2str(k) 'th iteration with residual ' num2str(res(k+1)*1000) ', because res increased'];
        disp(text)
        break
    end
    
end
end



function [ U, W, V, res, W_old, V_old ] = UpdateWandVStep2( Y1, Y2, U, V, W, F, par, net, Iter_count, W_old, V_old )
% Solving eq. (7) with a projected gradient descent method.

maxIter = 1;
epsilon = 1.01;
gamma2 = 1.01;
eta = 1;

global sigmas

res(1) = norm(Y1-U*W,'fro')^2+norm(Y2-F*U*V,'fro')^2+100;

for k=1:maxIter
    V1 = V;
    W1 = W;
    
    % 2.2. Update the AbundancesD
    ek = gamma2 * norm( F*U*U'*F' ,'fro');
    V = (Iter_count-1)/(Iter_count+2)*(V-V_old) + V;
    V_old = V1;
    W_old = W1;
    P = V - 1/ek * U' * F' * ( F*U*V - Y2 );
    
    V = Pplusb(P);
%     V = P;
%     V(V<0) = 0;
%     V(V>1) = 1;
    
    % Uncomment Tau_multi and comment the following line to use the sparse
    % constraint
    
    V = reshape(V,[size(V,1),size_msi(1),size_msi(2)]);
    V = permute(V,[2,3,1]);
    W = imfilter(V, Blur, 'circular');
    W = W(start_pos(1):(par.size_msi(1)/par.size_hsi(1)):end, par.start_pos(2):(par.size_msi(2)/par.size_hsi(2)):end,:);
    W = permute(W,[3,1,2]);
    V = permute(V,[3,1,2]);
    W = reshape(W,[size(W,1),size_hsi(1)*size_hsi(2)]);
    V = reshape(V,[size(V,1),size_msi(1)*size_msi(2)]);
    
    % Calculation of residuals
    res(k+1,1) = norm(Y1-U*W,'fro')^2+norm(Y2-F*U*V,'fro')^2;
    
    % Checks for exiting iteration
    if (1/ (k+1) * res(k)) < epsilon
        fprintf(['Multi: Iter ' num2str(k) ', res: ' num2str(res(k+1)*1000) '. '])
        break
    end
    
    if (res(k+1) / res(k))>1
        W = W_old;
        V = V_old;
        text = ['Multi: Exited during ' num2str(k) 'th iteration with residual ' num2str(res(k+1)*1000) ', because res increased'];
        disp(text)
        break
    end
    
end
end


function V = Pplusb(V)
% Simplex Projection

V = hyperConvert3d(V,2);
V1 = reproject_simplex_mex_fast(V);
V = hyperConvert2d(V1);

end


function U = Tau_multi(U,s)

% keep only the first s largest entries of U
U1 = reshape(U,[],1);
[values, ind] = sort(U1,'descend');
U1 = zeros(length(U1),1);
U1(ind(1:s),1) = values(1:s);
U = reshape(U1,size(U));

end




% -------------------------------------------------------------------------
function net = simpleMergeBatchNorm(net)
% -------------------------------------------------------------------------

for l = 1:numel(net.layers)
  if strcmp(net.layers{l}.type, 'bnorm')
    if ~strcmp(net.layers{l-1}.type, 'conv')
      error('Batch normalization cannot be merged as it is not preceded by a conv layer.') ;
    end
    [filters, biases] = mergeBatchNorm(...
      net.layers{l-1}.weights{1}, ...
      net.layers{l-1}.weights{2}, ...
      net.layers{l}.weights{1}, ...
      net.layers{l}.weights{2}, ...
      net.layers{l}.weights{3}) ;
    net.layers{l-1}.weights = {filters, biases} ;
  end
end
end

% -------------------------------------------------------------------------
function [filters, biases] = mergeBatchNorm(filters, biases, multipliers, offsets, moments)
% -------------------------------------------------------------------------
% wk / sqrt(sigmak^2 + eps)
% bk - wk muk / sqrt(sigmak^2 + eps)
a = multipliers(:) ./ moments(:,2) ;
b = offsets(:) - moments(:,1) .* a ;
biases(:) = biases(:) + b(:) ;
sz = size(filters) ;
numFilters = sz(4) ;
filters = reshape(bsxfun(@times, reshape(filters, [], numFilters), a'), sz) ;

end

function net = simpleRemoveLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = simpleFindLayersOfType(net, type) ;
net.layers(layers) = [] ;

end

% -------------------------------------------------------------------------
function layers = simpleFindLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = find(cellfun(@(x)strcmp(x.type, type), net.layers)) ;
end



