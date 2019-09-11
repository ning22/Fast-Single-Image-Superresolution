function [Xest,FX] = INVLS(FB,FBC,F2B,FR,tau,Nb,nr,nc,m,varargin)
%**************************************************************************
% Author: Ningning ZHAO (2015 Oct.)
% University of Toulouse, IRIT/INP-ENSEEIHT
% Email: buaazhaonn@gmail.com
%
% USAGE: Analytical solution as below
%        x = (B^H S^H SH + tau I )^(-1) R
% INPUT:
    %  FB-> Fourier transform of the blurring kernel B
    %  FBC->conj(FB)
    %  F2B->abs(FB)^2
    %  FR-> Fourier transform of R
    %  Nb-> scale factor Nb = dr*dc
    %  nr,nc-> size of the observation
    %  m-> No. of the pixels of the observation m = nr*nc    
% OUTPUT:
    %  Xest->Analytical solution
    %  FX->Fourier transform of the analytical solution
%************************************************************************** 
if nargin ==9
    F2D = 1;  
elseif nargin==10
    F2D = varargin{1}; % TV prior: F2D = F2DH + F2DV +c
end

x1 = FB.*FR./F2D;
FBR = BlockMM(nr,nc,Nb,m,x1);
invW = BlockMM(nr,nc,Nb,m,F2B./F2D);
invWBR = FBR./(invW + tau*Nb);

fun = @(block_struct) block_struct.data.*invWBR;
FCBinvWBR = blockproc(FBC,[nr,nc],fun);
FX = (FR-FCBinvWBR)./F2D/tau;
Xest = real(ifft2(FX));


function x = BlockMM(nr,nc,Nb,m,x1)
myfun = @(block_struct) reshape(block_struct.data,m,1);
x1 = blockproc(x1,[nr nc],myfun);
x1 = reshape(x1,m,Nb);
x1 = sum(x1,2);
x = reshape(x1,nr,nc);


