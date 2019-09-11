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

rng(0,'v5normal');
B = fspecial('gaussian',7,3);
[FB,FBC,F2B,Bx] = HXconv(refl,B,'Hx');
N = numel(refl);
Psig  = norm(Bx,'fro')^2/N;
BSNRdb = 40;
sigma = norm(Bx-mean(mean(Bx)),'fro')/sqrt(N*10^(BSNRdb/10));
rng(0,'v5normal');
y = Bx + sigma*randn(size(refl));
FY = fft2(y);
sig2n = sigma^2;

d = 2;
dr = d;
dc = d;
Nb = dr*dc;
y = y(1:dr:end,1:dc:end);
reflp = refl(1:dr:end,1:dc:end);
yinp = imresize(y,d,'bicubic');

%% L2 norm Super-resolution with Analytical solution
taup = 2e-3;
tau = taup*sig2n;
[nr,nc] = size(y);
m = nr*nc;
nrup = nr*d;
ncup = nc*d;

% xp= refl;
xp = yinp; 
STy = zeros(nrup,ncup);
STy(1:d:end,1:d:end)=y;
FR = FBC.*fft2(STy) + fft2(2*tau*xp);

tic
Xest_analytic = INVLS(FB,FBC,F2B,FR,2*tau,Nb,nr,nc,m);
toc
 
ISNR_analytic = ISNR_cal(yinp,refl,Xest_analytic);
fprintf('Analytic: ISNR = %g\n',ISNR_analytic);

clf, imagesc(Xest_analytic); colormap gray; axis off
title('Analytical Solution');
%% L2 norm Super-resolution with direct ADMM
stoppingrule = 1;
tolA = 1e-4;
maxiter = 200;
X = yinp;
BX = ifft2(FB.*fft2(X));
resid =  ifft2(y - BX(1:dr:end,1:dc:end));
prev_f = 0.5*(resid(:)'*resid(:)) + tau*sum(abs(X(:)-xp(:)));
distance = zeros(1,maxiter);
criterion = zeros(1,maxiter);
mses = zeros(1,maxiter);

muVec = 0.06;  %muVec = linspace(0.01,0.05,5);
nt= numel(muVec);
objective = zeros(nt,maxiter);
objective(:,1) = prev_f;
ISNR_admmSet = zeros(nt,maxiter);
[nr,nc] = size(y);
nrup = nr*dr;
ncup = nc*dc;
STytemp = ones(nrup,ncup);
STytemp(1:dr:end,1:dc:end)=y;
ind1 = find((STytemp-STy)==1);
ind2 = find((STytemp-STy)==0);
for t = 1:nt
    mu = muVec(t);
    gam = tau/mu;
    X = yinp;
    U = X;
    D = zeros(nrup,ncup);
tic
for i = 1:maxiter
    Xold = X;
   % update x
    V = U-D;
    Fb = FBC.*fft2(V) + fft2(2*gam*xp);
    FX = Fb./(F2B + 2*gam);
    X = ifft2(FX);
    
    % update u
    rr = mu*(ifft2(FB.*FX) + D);   
    temp1 = rr(ind1)./mu; 
    temp2 = (rr(ind2) + STy(ind2))./(1+mu);
    U(ind1) = temp1;
    U(ind2) = temp2;
   % update d
    D = D + (ifft2(FB.*FX)-U);
 %%
    ISNR_admmSet(t,i) = ISNR_cal(yinp,refl,X);       
    BX = ifft2(FB.*fft2(X));
    resid =  ifft2(y - BX(1:dr:end,1:dc:end));
    objective(t,i+1) = 0.5*(resid(:)'*resid(:)) + tau*sum(abs(X(:)-xp(:)));
    distance(i) = norm(X(:)-U(:),2);
    err = X-true;
    mses(i) =  (err(:)'*err(:))/N;
    
    switch stoppingrule
        case 1
            criterion(i) = abs(objective(t,i+1)-objective(t,i))/objective(t,i);
        case 2
            criterion(i) = distance(i);
    end
    
     if ( criterion(i) < tolA )
         break
     end
end  
    toc
    ISNR_admm = ISNR_cal(yinp,refl,X);
    fprintf('ADMM: Iter = %d, mu = %g, ISNR = %g\n',...
        i,muVec(t), ISNR_admm);
end

Xest_ADMM = X;   

clf, imagesc(Xest_ADMM); colormap gray; axis off
title('direct ADMM');

