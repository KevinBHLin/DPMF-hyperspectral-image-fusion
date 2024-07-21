function [U, W, Eng] = LSMMUnmixing(HSI, sigma_hsi, rank, truth, fixed, SNRh)

    if nargin<5
        fixed = 0;
    end
    maxIter = 1500;
    epsilon = 0.0001;
    resAB(1) = 1;
    s2 = rank/2;
%     s2 = rank;
    
    displayerror = 0;
    
    fprintf(['Initialization before spectral unmixing...\n'])
    
    % Initialisations
    Y = hyperConvert2d(HSI);
    if displayerror==1
        truth = hyperConvert2d(truth);
    end
    U = sisal(Y,rank, 'spherize', 'no','MM_ITERS',100, 'TAU', 0.01, 'verbose',0);%0.02
%     [U, ~, ~ ]= VCA(Y,'Endmembers',rank,'SNR',SNRh);
    %U(U<0) = 0;
    W = sunsal(U, Y,'POSITIVITY','yes','ADDONE','yes');
    

    Y = bsxfun(@rdivide,Y,sigma_hsi);
    U = bsxfun(@rdivide,U,sigma_hsi);

    U_old = U;
    W_old = W;
    text = ['Begin spectral unmixing...'];
    disp(text)
    for j=1:maxIter

        % hyperspectral least-squares
        [U, ~, res,U_old] = lowResStep(Y,U,W,sigma_hsi,j,U_old);
        resA2(j) = min(res);

        % multispectral least-squares
        [~, W, res,W_old] = highResStep(Y,U,W,s2,j,W_old);
        resB2(j) = min(res);

        % Residual of the objective function (5a)
        resAB(j+1) = resB2(j);

        if displayerror==1
            % Compute RMSE only for printing during procedure
            RMSE(j) = hyperErrRMSE(truth,bsxfun(@times,U,sigma_hsi)*W);
        end

        % Convergence checks
        if ( resAB(j) / resAB(j+1) ) > 1+epsilon || ( resAB(j) / resAB(j+1) ) < 1-epsilon
            if displayerror==1
                fprintf(['Iter: ' num2str(j) ' RMSE: ' num2str(RMSE(j)) '\n'])
            end
        else
            if displayerror==1
                fprintf(['Iter: ' num2str(j) ' RMSE: ' num2str(RMSE(j)) '\n'])
                fprintf(['Stopped after ' num2str(j) ' iterations. Final RMSE: ' num2str(RMSE(j)) '\n'])
            end
            break
        end


    end
    U = bsxfun(@times,U,sigma_hsi);
    
    if fixed==0
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        EngImg = zeros(size(U,2),size(HSI,3),size(HSI,1)*size(HSI,2));
        Eng = zeros(size(U,2),1);
        for i = 1:size(U,2)
            EngImg(i,:,:) = U(:,i)*W(i,:);
            Eng(i) = sum(EngImg(i,:,:),'all');
        end

        [C,I] = sort(Eng);
        removeI = I(cumsum(C)./sum(C)<0.1);
        U(:,removeI) = [];
        W(removeI,:) = [];

        rank = size(U,2);
        s2 = rank;%rank/2;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        U = bsxfun(@rdivide,U,sigma_hsi);

        fprintf(['Repeat spectral unmixing based on the estimated number of endmembers (rank = ' num2str(rank) ')...\n'])

        resAB(1) = 1;
        U_old = U;
        W_old = W;
        for j=1:maxIter

            % hyperspectral least-squares
            [U, ~, res,U_old] = lowResStep(Y,U,W,sigma_hsi,j,U_old);
            resA2(j) = min(res);

            % multispectral least-squares
            [~, W, res,W_old] = highResStep(Y,U,W,s2,j,W_old);
            resB2(j) = min(res);

            % Residual of the objective function (5a)
            resAB(j+1) = resB2(j);

            if displayerror==1
                % Compute RMSE only for printing during procedure
                RMSE(j) = hyperErrRMSE(truth,bsxfun(@times,U,sigma_hsi)*W);
            end

            % Convergence checks
            if ( resAB(j) / resAB(j+1) ) > 1+epsilon || ( resAB(j) / resAB(j+1) ) < 1-epsilon
                if displayerror==1
                    fprintf(['Iter: ' num2str(j) ' RMSE: ' num2str(RMSE(j)) '\n'])
                end
            else
                if displayerror==1
                    fprintf(['Iter: ' num2str(j) ' RMSE: ' num2str(RMSE(j)) '\n'])
                    fprintf(['Stopped after ' num2str(j) ' iterations. Final RMSE: ' num2str(RMSE(j)) '\n'])
                end
                break
            end


        end
        U = bsxfun(@times,U,sigma_hsi);
    end
    
    fprintf(['End spectral unmixing\n'])
end

function [ E, A, res, A_old ] = highResStep( M, E, A, sparse_factor, Iter_count, A_old )
% Solving eq. (7) with a projected gradient descent method.

maxIter = 1;
epsilon = 1.01;
gamma2 = 1.01;

N = size(M, 2);
beta = round(sparse_factor*N); % Number of desirable non-zero entries

res(1) = norm(M-E*A,'fro')+100;

for k=1:maxIter
    A1 = A;
    
    % 2.2. Update the Abundances
    dk = gamma2 * norm( E*E' ,'fro');
    A = (Iter_count-1)/(Iter_count+2)*(A-A_old) + A;
    A_old = A1;
    V = A - 1/dk * E' * ( E*A - M );
    
    % Uncomment Tau_multi and comment the following line to use the sparse
    % constraint
    % A = Tau_multi(Pplusb(V),beta);
    A = Pplusb(V);
    
    % Calculation of residuals
    res(k+1,1) = sqrt(norm(M-E*A,'fro')^2/size(M,1)/size(M,2));
    
    % Checks for exiting iteration
    if (1/res(k+1) * res(k)) < epsilon
        fprintf(['Multi: Iter ' num2str(k) ', res: ' num2str(res(k+1)*1000) '. '])
        break
    end
    
    if (res(k+1) / res(k))>1
        A = A_old;
        text = ['Multi: Exited during ' num2str(k) 'th iteration with residual ' num2str(res(k+1)*1000) ', because res increased'];
        disp(text)
        break
    end
    
end
end

function [ E, A, res, E_old ] = lowResStep( H, E, A, th, Iter_count, E_old )
% Solving eq. (6) with a projected gradient descent method.

maxIter = 1;
epsilon = 1.01;
gamma1 = 1.01;

notfirst = 0;

res(1) = norm(H-E*A,'fro')+100;

for k=1:maxIter
    E1 = E;
    
    % 2.1. Update of signatures
    ck = gamma1 * norm(A*A','fro');
    E = (Iter_count-1)/(Iter_count+2)*(E-E_old) + E;
    E_old = E1;
    U = E - 1/ck * ( E*A - H ) * A';
    E = Pplusa(U,th);
    
    % Calculation of residuals
    res(k+1,1) = sqrt(norm(H-E*A,'fro')^2/size(H,1)/size(H,2));
    
    % Checks for exiting iteration
    if (1/res(k+1) * res(k)) < epsilon
        fprintf(['Hyper: Iter ' num2str(k) ', res: ' num2str(res(k+1)*1000) '. '])
        break
    end
    
    if (res(k+1) / res(k))>1
        if notfirst == 1
            E = E_old;
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

function V = Pplusb(V)
% Simplex Projection

V = hyperConvert3d(V,2);
V1 = reproject_simplex_mex_fast(V);
V = hyperConvert2d(V1);

end