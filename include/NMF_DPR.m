function [U,V] = NMF_DPR(HSI, MSI, Truth, F, Blur, start_pos, sigma_hsi, sigma_msi, SNRm, SNRh)

% [FusionImage, BlurVersion, ResOutline, ResOutFrame, val_RMSE, RMSE_ITER, ConvegenIter, H, U_avr, W_avr, V_avr] = BSDMF_FAST( HSI, MSI, F, 0, 'SVD', 10, 30, 15, 20, Truth, 0);


maxIter = 150;
useGPU = 1;
%epsilon = 0.0001;
epsilon = 0.001;
rank = 16;
resAB(1) = 1;

[m,n,L] = size(HSI);
[M,N,l] = size(MSI);
size_hsi = [m,n];
size_msi = [M,N];

    load('NetworkModel\\NetworkTrainedGrayImage\\result.mat');
    disp(['The CNN is trained by Gray Images'])
 
    for i=1:size(net.layers,2)
        if(strcmp(net.layers{i}.type,'bnorm')|| strcmp(net.layers{i}.type,'conv'))
            net.layers{i}.opts={};
        end
        if(strcmp(net.layers{i}.type,'conv'))
            net.layers{i}.pad=double(net.layers{i}.pad);
            net.layers{i}.stride = double(net.layers{i}.stride);
            net.layers{i}.dilate = double(net.layers{i}.dilate);
        end
    end

    net = vl_simplenn_tidy(net);
    net.layers = net.layers(1:end-1);
    net = vl_simplenn_tidy(net);
    net = simpleMergeBatchNorm(net);
    net = simpleRemoveLayersOfType(net, 'bnorm');
    if useGPU
        net = vl_simplenn_move(net, 'gpu') ;
    end

%     Y = single(HSI);
%     Y = hyperConvert2d(Y);
%     Y = bsxfun(@rdivide, Y, sigma_hsi);
%     Y = hyperConvert3d(Y,m);
%     if useGPU
%         Y = gpuArray(Y);
%         for ch=1:size(Y,3)
%             imres = vl_simplenn(net,squeeze(Y(:,:,ch)),[],[],'conserveMemory',true,'mode','test');
%             Y(:,:,ch) = squeeze(Y(:,:,ch)) + imres(end).x;
%         end
%         Y = gather(Y);
%     else
%         for ch=1:size(Y,3)
%             imres = vl_simplenn(net,squeeze(Y(:,:,ch)),[],[],'conserveMemory',true,'mode','test');
%             Y(:,:,ch) = squeeze(Y(:,:,ch)) + imres(end).x;
%         end
%     end
%     Y = hyperConvert2d(Y);
%     Y = double(bsxfun(@times, Y, sigma_hsi));
%     Y(Y<0) = 0; Y(Y>1) =1;
%    
% %     [U, indicies] = vca(Y, rank);
%     U = sisal(Y, rank, 'spherize', 'no','MM_ITERS',80, 'TAU', 0.006, 'verbose',0);
% %     U(U<0) = 0; U(U>1) = 1;
%     W = sunsal(U, Y,'POSITIVITY','yes','ADDONE','yes');
% %     stdW = std(W');
% %     W = bsxfun(@rdivide,W,stdW');
% %     U = bsxfun(@times,U,stdW);
%     V = hyperConvert2d(imresize(hyperConvert3d(W,m),[M,N]));
%     
    
%     [U,V] = LSMMUnmixing(Truth, 1, rank, Truth);
%     W = hyperConvert2d(imresize(hyperConvert3d(V,M),[m,n]));
    [U,W] = LSMMUnmixing(HSI, sigma_hsi, rank, imresize(Truth,[m,n]),1, SNRh);
%     Y = hyperConvert2d(HSI);
%     U = sisal(Y,rank, 'spherize', 'no','MM_ITERS',80, 'TAU', 0.006, 'verbose',0);
%     W = sunsal(U, Y,'POSITIVITY','yes','ADDONE','yes');
    V = hyperConvert2d(imresize(hyperConvert3d(W,m),[M,N]));
    
Y1 = hyperConvert2d(HSI);
Y2 = hyperConvert2d(MSI);
Truth = hyperConvert2d(Truth);

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

U = bsxfun(@rdivide,U,sigma_hsi);
F = bsxfun(@rdivide,F,sigma_msi);
F = bsxfun(@times,F',sigma_hsi)';

resAB(1) =  norm(Y1-U*W,'fro')^2+norm(Y2-F*U*V,'fro')^2;

RMSE(1) = hyperErrRMSE(Truth, bsxfun(@times,U*V,sigma_hsi));
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

resAB(1) =  norm(Y1-U*W,'fro')^2+norm(Y2-F*U*V,'fro')^2;
RMSE(1) = hyperErrRMSE(Truth, bsxfun(@times,U*V,sigma_hsi));
fprintf(['Iter: ' num2str(0) ' RMSE: ' num2str(RMSE(1)) '\n'])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

U_old = U;
W_old = W;
V_old = V;

for j=1:maxIter
%     W = W.*sqrt(norm( U*U' ,'fro'));
%     V = V.*sqrt(norm( F*U*U'*F' ,'fro'));
%     U = U./sqrt(norm( U*U' ,'fro'));
%     F = F.*sqrt(norm( U*U' ,'fro'))./sqrt(norm( F*U*U'*F' ,'fro'));

    % update the abundance matrix W and V
    [~, W, V, res, W_old, V_old] = UpdateWandVStep(Y1, Y2, U, V, W, F, Blur, start_pos, sigma_hsi, sigma_msi, net, size_hsi, size_msi, useGPU, j, W_old, V_old);
    resB2(j) = min(res);
    
    resAB(j+1) =  norm(Y1-U*W,'fro')^2+norm(Y2-F*U*V,'fro')^2; %resA2(j)+resB2(j);
    
    % update the spectral matrix U
    [U, ~, ~, res, U_old] = UpdateUStep(Y1, Y2, U, V, W, F, sigma_hsi, sigma_msi, Uth, j, U_old);
    resA2(j) = min(res);
    
    % Residual of the objective function (5a)
    resAB(j+1) =  norm(Y1-U*W,'fro')^2+norm(Y2-F*U*V,'fro')^2; %resA2(j)+resB2(j);
    
    % Compute RMSE only for printing during procedure
    RMSE(j) = hyperErrRMSE(Truth, bsxfun(@times,U*V,sigma_hsi));
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

U = bsxfun(@times,U,sigma_hsi);

end


function [ U, W, V, res, U_old] = UpdateUStep( Y1, Y2, U, V, W, F, sigma_hsi, sigma_msi, th, Iter_count, U_old )
% Solving eq. (6) with a projected gradient descent method.

maxIter = 1;
epsilon = 1.01;
gamma1 = 1.01;

notfirst = 0;

res(1) = norm(Y1-U*W,'fro')+100;

D = zeros(size(V,1),size(V,1)*(size(V,1)-1)/2);
point = 1;
for i=1:(size(V,1)-1)
    for j=i+1:size(V,1)
        D(i,point) = 1;
        D(j,point) = -1;
        point = point+1;
    end
end
DD = D*D';
lambda = 8;

for k=1:maxIter
    U1 = U;
    
    % 2.1. Update of signatures
%     ck = gamma1 * norm(W*W','fro') + gamma1 * norm(V*V','fro');
%     E = U - 1/ck * ( U*W - Y1 ) * W' - 1/ck * F' * ( F*U*V - Y2 ) * V';
%     ck = gamma1 * norm(W*W'+lambda*DD,'fro');
    ck = gamma1 * norm(W*W','fro');
    U = (Iter_count-1)/(Iter_count+2)*(U-U_old) + U;
    U_old = U1;
%     E = U - 1/ck * ( U*W - Y1 ) * W' - 1/ck * U*DD;
    E = U - 1/ck * ( U*W - Y1 ) * W';
    U = Pplusa(E, th);
    
    % Calculation of residuals
    res(k+1,1) = sqrt(norm(Y1-U*W,'fro')^2/size(Y1,1)/size(Y1,2));
    
    % Checks for exiting iteration
    if (1/res(k+1) * res(k)) < epsilon
        fprintf(['Hyper: Iter ' num2str(k) ', res: ' num2str(res(k+1)*1000) '. '])
        break
    end
    
    if (res(k+1) / res(k))>1
        if notfirst == 1
            U = U_old;
            text = ['Hyper: Exited during ' num2str(k) 'th iteration with residual ' num2str(res(k+1)*1000) ', because res increased'];
            disp(text)
            break
        else
            notfirst = 1;
        end
    end
    
end
end

function U = Pplusa(U, th)
% max{0,U}
U(U<0) = 0;
U = bsxfun(@times,U,th);
U(U>1) = 1;
U = bsxfun(@rdivide,U,th);
end

function [ U, W, V, res, W_old, V_old ] = UpdateWandVStep( Y1, Y2, U, V, W, F, Blur, start_pos, sigma_hsi, sigma_msi, net, size_hsi, size_msi, useGPU, Iter_count, W_old, V_old )
% Solving eq. (7) with a projected gradient descent method.

maxIter = 1;
epsilon = 1.01;
gamma2 = 1.01;
eta = 0.9;

res(1) = norm(Y1-U*W,'fro')^2+norm(Y2-F*U*V,'fro')^2+100;

for k=1:maxIter
    V1 = V;
    W1 = W;
    
    % 2.2. Update the Abundances
    ek = gamma2 * norm( F*U*U'*F' ,'fro');
    V = (Iter_count-1)/(Iter_count+2)*(V-V_old) + V;
    V_old = V1;
    W_old = W1;
    P = V - 1/ek * U' * F' * ( F*U*V - Y2 );
    
    noise_p_sigma = sqrt(1/ek)*eta;
    
    % Uncomment Tau_multi and comment the following line to use the sparse
    % constraint
    P = P./noise_p_sigma;
    P = reshape(P,[size(P,1),size_msi(1),size_msi(2)]);
    
    P = single(P);
    if useGPU 
        P = gpuArray(P);
        for ch=1:size(P,1)
            imres = vl_simplenn(net,squeeze(P(ch,:,:)),[],[],'conserveMemory',true,'mode','test');
            P(ch,:,:) = squeeze(P(ch,:,:)) + imres(end).x;
        end
        V = gather(P);
    else
        for ch=1:size(P,1)
            imres = vl_simplenn(net,squeeze(P(ch,:,:)),[],[],'conserveMemory',true,'mode','test');
            P(ch,:,:) = squeeze(P(ch,:,:)) + imres(end).x;
        end
        V = P;
    end
    V = double(V);
    V = V.*noise_p_sigma;
    V(V<0) = 0; V(V>1) = 1;
    V = permute(V,[2,3,1]);
%     W = gaussian_down_sample(V,size_msi(1)/size_hsi(1));
    W = imfilter(V, Blur, 'circular');
    W = W(start_pos(1):(size_msi(1)/size_hsi(1)):end, start_pos(2):(size_msi(2)/size_hsi(2)):end,:);
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



% function [W, V] = Pplusb(Q, P, model, sigma_hsi, sigma_msi, net_enable)
% % Simplex Projection
% 
% % V = hyperConvert3d(V,2);
% % V1 = reproject_simplex_mex_fast(V);
% % V = hyperConvert2d(V1);
% if net_enable==1
%     W = zeros(size(Q));
%     V = zeros(size(P));
%     for i=1:size(Q,1)
%         result = cell(py.test.UseModel(model, P(i,:,:),Q(i,:,:)));
%         W(i,:,:) = double(result{2});
%         V(i,:,:) = double(result{1});
%     end
%     W(W<0)=0;
%     V(V<0)=0;
% else
%     W = Q; W(W<0)=0;
%     V = P; V(V<0)=0;
% end
% 
% end

% function U = Tau_multi(U,s)
% 
% % keep only the first s largest entries of U
% U1 = reshape(U,[],1);
% [values, ind] = sort(U1,'descend');
% U1 = zeros(length(U1),1);
% U1(ind(1:s),1) = values(1:s);
% U = reshape(U1,size(U));
% 
% end




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



