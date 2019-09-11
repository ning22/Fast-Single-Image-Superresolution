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

%% TV norm Super resolution with direct ADMM
taup = 1e-2;
tau  = taup*sig2n; 
[nr,nc] = size(y);
nrup = nr*d;
ncup = nc*d;
Nb = dr*dc;
m = nr*nc;

stoppingrule = 1;
tolA = 1e-3;
% tolA = -inf;
maxiter = 800;
[nr,nc] = size(y);
distance = zeros(2,maxiter);
distance_max = zeros(1,maxiter);
criterion = zeros(1,maxiter);
mses = zeros(1,maxiter);

muSet = 0.005;   % muSet = linspace(0.006,0.01,5);  
gamSet = tau./muSet;
nt= numel(muSet);
objective = zeros(nt,maxiter);
ISNR_admmSet = zeros(nt,maxiter);
times = zeros(1,maxiter);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define the difference operator kernel
dh = zeros(nrup,ncup);
dh(1,1) = 1;
dh(1,2) = -1;
% 
dv = zeros(nrup,ncup);
dv(1,1) = 1;
dv(2,1) = -1;
% 
% compute FFTs for filtering
FDH = fft2(dh);
FDHC = conj(FDH);
F2DH = abs(FDH).^2;
FDV = fft2(dv);
FDVC = conj(FDV);
F2DV = abs(FDV).^2;
c = 1e-8;
F2D = F2DH + F2DV +c;

STy = zeros(nrup,ncup);
STy(1:d:end,1:d:end)=y;
STytemp = ones(nrup,ncup);
STytemp(1:dr:end,1:dc:end)=y;
ind1 = find((STytemp-STy)==1);
ind2 = find((STytemp-STy)==0);
FBTSTy = FBC.*fft2(STy);

X = yinp;
FX = fft2(X);
BX = ifft2(FB.*fft2(X));
resid =  y - BX(1:dr:end,1:dc:end);
TVpenalty = sum(sum(sqrt(abs(ifft2(FDH.*(FX))).^2+abs(ifft2(FDV.*(FX))).^2)));
prev_f = 0.5*(resid(:)'*resid(:)) + tau*TVpenalty;
objective(:,1) = prev_f;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t = 1:nt
    mu = muSet(t);
    X = yinp;
    U1 = X;
    U2 = X;
    U3 = X;
    D1 = 0*X;
    D2 = D1;
    D3 = D1;
    
    I_DH = FDHC ./(F2DH + F2DV + F2B);
    I_DV = FDVC ./(F2DH + F2DV + F2B);
    I_BB = FBC  ./(F2DH + F2DV + F2B);
    tic
for i = 1:maxiter
    ISNR_admmSet(t,i) = ISNR_cal(yinp,refl,X); 
   %%%%%%%%%%%%%%%%%%%%%%%%%% update X %%%%%%%%%%%%%%%%%%%%%%%%%%%
   % argmin_x  mu/2*||HX  - U1 + D1||^2 + 
   %           mu/2*||DhX - U2 + D2||^2 + 
   %           mu/2*||DvX - U3 + D3||^2
    V1 = U1-D1;
    V2 = U2-D2;
    V3 = U3-D3;
    FX = I_BB.*fft2(V1) + I_DH.*fft2(V2) + I_DV.*fft2(V3);
    
%     FR = FBC.*fft2(V1) + FDHC.*fft2(V2) + FDVC.*fft2(V3);
%     FX = FR./(F2DH + F2DV + F2B);
    X = ifft2(FX);

   %%%%%%%%%%%%%%%%%%%%%%%%%% update U %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % argmin_u1 0.5||Su1 - y||_1 + mu/2*||HX - U1 + D1||^2 
    HX = ifft2(FB.*FX);
    rr = mu*(HX + D1);
    temp1 = rr(ind1)./mu; 
    temp2 = (rr(ind2) + STy(ind2))./(1+mu);
    U1(ind1) = temp1;
    U1(ind2) = temp2;
   % argmin_u(2,3) tau*||sqrt(U2^2 + U3^2)||_1 + 
   %          mu/2*||DhX - U2 + D2||^2 +
   %          mu/2*||DvX - U3 + D3||^2
    DhX = ifft2(FX.*FDH);
    DvX = ifft2(FX.*FDV);
    NU1 = DhX + D2;
    NU2 = DvX + D3;
    NU = sqrt(NU1.^2+NU2.^2);
    A = max(0, NU-gamSet(t));
    A = A./(A + gamSet(t));
    U2 = A.*NU1;
    U3 = A.*NU2;
   %%%%%%%%%%%%%%%%%%%%%%%%%% update D %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D1 = D1 + (HX-U1);
    D2 = D2 + (DhX-U2);
    D3 = D3 + (DvX-U3);
 %%   
    resid =  y - HX(1:dr:end,1:dc:end);
    TVpenalty = sum(sum(sqrt(abs(ifft2(FDH.*(FX))).^2+abs(ifft2(FDV.*(FX))).^2)));
    objective(t,i+1) = 0.5*(resid(:)'*resid(:)) + tau*TVpenalty;
    distance(1,i) = norm(DhX(:)-U1(:),2);
    distance(2,i) = norm(DvX(:)-U2(:),2);
    distance_max(i)=distance(1,i)+distance(2,i);
    err = X-refl;
    mses(i) =  (err(:)'*err(:))/N;
    times(i) = toc;
    switch stoppingrule
        case 1      
            criterion(i) = abs(objective(t,i+1)-objective(t,i))/objective(t,i);
        case 2
            criterion(i) = distance_max(i);
    end

     if ( criterion(i) < tolA )
         break
     end
end
    toc
    ISNR_admm  = ISNR_cal(yinp,refl,X);
    fprintf('direct ADMM: mu = %g, Iter = %d, ISNR = %g\n',...
    muSet(t), i, ISNR_admm);
end
clf, imagesc(X); colormap gray; axis off
title('direct ADMM');

%% Super resolution with FSR-ADMM
for t = 1:nt
    mu = muSet(t);
    X = yinp;
    U1 = X;
    U2 = X;
    D1 = 0*X;
    D2 = D1;
    
    tic
    time0 = tic;
for i = 1:maxiter

    ISNR_admmSet(t,i) = ISNR_cal(yinp,refl,X); 
   %%%%%%%%%%%%%%%%%%%%%%%%%% update X %%%%%%%%%%%%%%%%%%%%%%%%%%%
   % argmin_x  .5*||y-SHx||^2 + 
   %           mu/2*||DhX - U1 + D1||^2 + 
   %           mu/2*||DvX - U2 + D2||^2
    V1 = U1-D1;
    V2 = U2-D2;
    FV1 = mu*FDHC.*fft2(V1);
    FV2 = mu*FDVC.*fft2(V2) ;
    FR = FBTSTy + FV1 + FV2;
    [X,FX] = INVLS(FB,FBC,F2B,FR,mu,Nb,nr,nc,m,F2D);
   %%%%%%%%%%%%%%%%%%%%%%%%%% update U %%%%%%%%%%%%%%%%%%%%%%%%%%%
   % argmin_u tau*||sqrt(U1^2 + U2^2)||_1 + 
   %          mu/2*||DhX - U1 + D1||^2 +
   %          mu/2*||DvX - U2 + D2||^2   
    DhX = ifft2(FX.*FDH);
    DvX = ifft2(FX.*FDV);
    NU1 = DhX + D1;
    NU2 = DvX + D2;
    NU = sqrt(NU1.^2+NU2.^2);
    A = max(0, NU-gamSet(t));
    A = A./(A + gamSet(t));
    U1 = A.*NU1;
    U2 = A.*NU2;
   %%%%%%%%%%%%%%%%%%%%%%%%%% update D %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D1 = D1 + (DhX-U1);
    D2 = D2 + (DvX-U2);
 %%
    
    BX = ifft2(FB.*FX);
    resid =  y - BX(1:dr:end,1:dc:end);
    TVpenalty = sum(sum(sqrt(abs(ifft2(FDH.*(FX))).^2+abs(ifft2(FDV.*(FX))).^2)));
    objective(t,i+1) = 0.5*(resid(:)'*resid(:)) + tau*TVpenalty;
    distance(1,i) = norm(DhX(:)-U1(:),2)^2;
    distance(2,i) = norm(DvX(:)-U2(:),2)^2;
    distance_max(i)=distance(1,i)+distance(2,i);
    err = X-refl;
    mses(i) =  (err(:)'*err(:))/N;
    times(i) = +toc;
    
    switch stoppingrule
        case 1      
            criterion(i) = abs(objective(t,i+1)-objective(t,i))/objective(t,i);
        case 2
            criterion(i) = distance_max(i);
    end

     if ( criterion(i) < tolA )
         break
     end
end
    toc
    ISNR_admm  = ISNR_cal(yinp,refl,X);
        fprintf('FSR-ADMM: mu = %g, Iter = %d, ISNR = %g\n',...
        muSet(t), i, ISNR_admm);
end

clf, imagesc(X); colormap gray; axis off
title('FSR ADMM');