function [Xest,WXest] = INVLS_WT(FB,FBC,F2B,WT,WR,tau,Nb,m,nr,nc,varargin)
%**************************************************************************
% Author: Ningning ZHAO (2015 Oct.)
% University of Toulouse, IRIT/INP-ENSEEIHT
% Email: buaazhaonn@gmail.com
%
% USAGE: Analytical solution as below
%        x = (W^H B^H S^H SHW + tau I )^(-1) R
% INPUT:
    %  FB-> Fourier transform of the blurring kernel B
    %  FBC-> conj(FB)
    %  F2B-> abs(FB)^2
    %  WT-> inverse wavelet transform operator
    %  FR-> Wavelet transform of R
    %  Nb-> scale factor Nb = dr*dc
    %  nr,nc-> size of the observation
    %  m-> No. of the pixels of the observation m = nr*nc    
% OUTPUT:
    %  Xest-> Analytical solution of the wavelet coefficients
    %  WXest-> Estimation of the image
%**************************************************************************  
if nargin ==10
    F2D = 1;
elseif nargin==11
    F2D = varargin{1}; % TV prior
end

FWR = fft2(WR);
x1 = FB.*FWR./F2D;
FBWR = BlockMM(nr,nc,Nb,m,x1);
invP = BlockMM(nr,nc,Nb,m,F2B./F2D);
invPWBR = FBWR./(invP + tau*Nb);

fun = @(block_struct) block_struct.data.*invPWBR;
FCBinvPWBR = blockproc(FBC,[nr,nc],fun);
FWX = (FWR-FCBinvPWBR)./F2D/tau;
WXest = real(ifft2(FWX));
Xest = WT(WXest);


function x = BlockMM(nr,nc,Nb,m,x1)
myfun = @(block_struct) reshape(block_struct.data,m,1);
x1 = blockproc(x1,[nr nc],myfun);
x1 = reshape(x1,m,Nb);
x1 = sum(x1,2);
x = reshape(x1,nr,nc);
