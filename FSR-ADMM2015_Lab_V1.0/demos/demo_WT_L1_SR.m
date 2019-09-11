clc
clear
close all

addpath ./../utils;
addpath ./../src;

%**************************************************************************
% Author: Ningning ZHAO (2015 Oct.)
% University of Toulouse, IRIT/INP-ENSEEIHT
% Email: buaazhaonn@gmail.com
%        nzhao@enseeiht.fr
% ---------------------------------------------------------------------
% Copyright (2015): Ningning Zhao, Qi Wei, Adrian Basarab, Denis Kouame and Jean-Yves Toureneret.
% 
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ---------------------------------------------------------------------
% 
% This set of MATLAB files contain an implementation of the algorithms
% described in the following paper:
% 
% [1] Fast Single Image Super-resolution. Ningning Zhao, Qi Wei, Adrian Basarab, Denis Kouame and Jean-Yves Toureneret.
%     [On line]http://arxiv.org/abs/1510.00143
% 
% [2] Qi Wei, Nicolas Dobigeon and Jean-Yves Tourneret, "Fast Fusion of Multi-Band Images Based on Solving a Sylvester Equation," 
%     IEEE Trans. Image Process., vol. 24, no. 11, pp. 4109-4121, Nov. 2015.
% 
% The code is available at http://zhao.perso.enseeiht.fr/
% 
% ---------------------------------------------------------------------
%**************************************************************************
%% Input Image
refl = double( imread('lena.bmp') );

B = fspecial('gaussian',9, 3);
[FB,FBC,F2B,Bx] = HXconv(refl,B,'Hx');
N = numel(refl);
Psig  = norm(Bx,'fro')^2/N;
BSNRdb = 40;
sigma = norm(Bx-mean(mean(Bx)),'fro')/sqrt(N*10^(BSNRdb/10));
rng(0,'v5normal');
y = Bx + sigma*randn(size(refl));

sig2n = sigma^2;
d = 4;
dr = d;
dc = d;
y = y(1:dr:end,1:dc:end);
yinp = imresize(y,d,'bicubic');

dr = d; dc = d;
[nr,nc] = size(y);
nrup = nr*d;
ncup = nc*d;
Nb = dr*dc;
m = nr*nc;
N = numel(yinp);

%% wavelet representation
levels = 4;
% wav = daubcqf(2); % Haar wavelet
% W = @(x) mirdwt_TI2D(x, wav, levels); % inverse transform (WT coeffs->image)
% WT = @(x) mrdwt_TI2D(x, wav, levels); % forward transform (image->WT coeffs)
% original = WT(refl);
% W = @(x)x;
% WT = @(x)x;
wave = 'db1';
[original,S] = wavedec2(refl,levels,wave);
WT = @(x) wavedec2(x, levels, wave); 
W = @(x) waverec2(x, S, wave); 

%% L1 norm SR (WT)
taup = 1e-3;
tau = taup*sig2n;
stoppingrule = 1;
tolA = 1e-3;
% tolA = -inf;
maxiter = 500;
distance = zeros(2,maxiter);
distance_max = zeros(1,maxiter);
criterion = zeros(1,maxiter);
mses = zeros(1,maxiter);
times = zeros(1,maxiter);

WTy = WT(yinp);  
WX = yinp;
WTX = WTy; 
BX = ifft2(FB.*fft2(WX));
resid =  y - BX(1:dr:end,1:dc:end);
prev_f = 0.5*(resid(:)'*resid(:)) + tau*sum(abs(WTX(:)));

STy = zeros(nrup,ncup);
STy(1:d:end,1:d:end)=y;
WBTSTy = real(ifft2(FBC.*fft2(STy)));

STytemp = ones(nrup,ncup);
STytemp(1:dr:end,1:dc:end)=y;
ind1 = find((STytemp-STy)==1);
ind2 = find((STytemp-STy)==0);

muSet = 0.05;  
gamSet = tau./muSet;
nt= numel(muSet);
objective = zeros(nt,maxiter);
ISNR_admmSet = zeros(nt,maxiter);
objective(:,1) = prev_f;

%% direct ADMM
for t = 1:nt
    mu = muSet(t);
    WX = yinp;
    X = WTy;
    U1 = WX;
    U2 = X;
    D1 = zeros(size(U1));
    D2 = zeros(size(X));
    
    I_BB = F2B + 1;
    tic
for i = 1:maxiter
    Xold = X;
    ISNR_admmSet(t,i) = ISNR_cal(yinp,refl,WX); 
   %%%%%%%%%%%%%%%%%%%% update x %%%%%%%%%%%%%%%%%%%%%%
   % argmin_x  mu/2*||HWX  - U1 + D1||^2 + 
   %           mu/2*||X - U2 + D2||^2 
    V1 = U1-D1;
    V2 = U2-D2;
    
    FWX = fft2(ifft2(FBC.*fft2(V1)) + W(V2))./I_BB;
    WX = ifft2(FWX);
    X = WT(WX);
    %%%%%%%%%%%%%%%%%%% update u %%%%%%%%%%%%%%%%%%%%%%
    % argmin_u1 0.5||Su1 - y||_1 + mu/2*||HWX - U1 + D1||^2 
    HWX = ifft2(FB.*FWX);
    rr = mu*(HWX + D1);
    temp1 = rr(ind1)./mu; 
    temp2 = (rr(ind2) + STy(ind2))./(1+mu);
    U1(ind1) = temp1;
    U1(ind2) = temp2;
    
    % argmin_u2 tau*||U2^2||_1 + 
    %          mu/2*||X - U2 + D2||^2 
    V = X + D2;
    U2 = max(abs(V)-gamSet(t),0).*sign(V); 
   %%%%%%%%%%%%%%%%%%%% update d %%%%%%%%%%%%%%%%%%%%%%
    D1 = D1 + (HWX-U1);
    D2 = D2 + (X-U2);
 %%
    
    BWX = ifft2(FB.*fft2(WX));
    resid =  y - BWX(1:dr:end,1:dc:end);     
    objective(t,i+1) = 0.5*(resid(:)'*resid(:)) + tau*sum(abs(X(:)));
    
    err = X-original;
    mses(i) =  (err(:)'*err(:))/N;

    times(i) = toc; 
    switch stoppingrule
        case 1
            criterion(i) = abs(objective(t,i+1)-objective(t,i))/objective(t,i);
        case 2
            distance(1,i) = norm(HWX(:)-U1(:),2);
            distance(2,i) = norm(X(:)-U2(:),2);
            distance_max(i)=distance(1,i)+distance(2,i);
            criterion(i) = distance_max(i);
    end
    
     if ( criterion(i) < tolA )
         break
     end
end
toc
    Xest = WX;
    ISNR_admm  = ISNR_cal(yinp,refl,Xest);
        fprintf('direct ADMM: mu = %g, Iter = %d, ISNR = %g\n',...
        muSet(t), i, ISNR_admm);
 
end
clf, imagesc(Xest); colormap gray; axis off
title('direct ADMM');
    

%% FSR-ADMM
for t = 1:nt
    mu = muSet(t);
    X = WTy;
    U = X;
    D = zeros(size(X));
    tic
for i = 1:maxiter
    ISNR_admmSet(t,i) = ISNR_cal(yinp,refl,WX);
   %%%%%%%%%%%%%%%%%%%% update x %%%%%%%%%%%%%%%%%%%%%%
   % argmin_x .5||SHWx - y||^2 + mu/2*||x-u+d||^2
    WR = WBTSTy + mu*W(U-D);
    [X,WX] = INVLS_WT(FB,FBC,F2B,WT,WR,mu,Nb,m,nr,nc);
    %%%%%%%%%%%%%%%%%%% update u %%%%%%%%%%%%%%%%%%%%%%
    % argmin_u tau*||u||_1 +  mu/2*||x-u+d||^2
    V = X + D;
    U = max(abs(V)-gamSet(t),0).*sign(V); 
   %%%%%%%%%%%%%%%%%%%% update d %%%%%%%%%%%%%%%%%%%%%%
    D = D + (X-U);
 %%
     
    BWX = ifft2(FB.*fft2(WX));
    resid =  y - BWX(1:dr:end,1:dc:end);     
    objective(t,i+1) = 0.5*(resid(:)'*resid(:)) + tau*sum(abs(X(:)));
    
    err = X-original;
    mses(i) =  (err(:)'*err(:))/N;
    times(i) = toc; 
    switch stoppingrule
        case 1
            criterion(i) = abs(objective(t,i+1)-objective(t,i))/objective(t,i);
        case 2
            distance(i) = norm(X(:)-U(:),2);
            criterion(i) = distance(i);
    end
     if ( criterion(i) < tolA )
         break
     end
end
    toc
    Xest = WX;
    ISNR_admm  = ISNR_cal(yinp,refl,Xest);
        fprintf('FSR-ADMM: mu = %g, Iter = %d, ISNR = %g\n',...
        muSet(t), i, ISNR_admm);
end

clf, imagesc(Xest); colormap gray; axis off
title('FSR-ADMM');