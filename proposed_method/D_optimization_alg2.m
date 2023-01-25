function [D, res] = D_optimization_alg2(D, S, X, A, B, N, opts)
%% implementation of the step for optimization of {d_k} in Algorithm 2



%% parameters

[H,W] = size(S);
[m,~,K] = size(D);

if nargin < 9
    opts = [];
end
if ~isfield(opts,'MaxIter')
    opts.MaxIter = 300;
end
if ~isfield(opts,'sig')
    opts.sig = 10;
end
if ~isfield(opts,'AutoSig')
    opts.AutoSig = 1;
end
if ~isfield(opts,'SigUpdateCycle')
    opts.SigUpdateCycle = 1;
end
if ~isfield(opts,'dcfilter')
    opts.dcfilter = 0;
end
if ~isfield(opts,'Vinit')
    opts.Vinit = zeros(H,W,K,'single');
end
if ~isfield(opts,'relaxParam')
    opts.relaxParam = 1.8;
end
if ~isfield(opts,'eAbs')
    opts.eAbs = 1e-4;
end
if ~isfield(opts,'eRel')
    opts.eRel = 1e-4;
end

%% initialization
alpha = opts.relaxParam;
Sf = fft2(S);
V = opts.Vinit;
sig = opts.sig;
SigUpdateCycle = opts.SigUpdateCycle;


eAbs = opts.eAbs;
eRel = opts.eRel;
eprid = 0;
eduad = 0;

MaxIter = opts.MaxIter;
r_cdl= inf; s_cdl = inf;
res.iterinf = [];
mu = 5; % varying rho parameter
tau = 1.2; % varying rho parameter

vec = @(x) x(:);
itr = 1;

%% CDL CYCLES
tsrt = tic;

Xf = fft2(X);
NX = abs(Xf).^2;
Xfcnj = conj(Xf);
b2 = B+conj(Xf).*Sf/N;

D = padarray(D,[H-m W-m],'post');
Nd = numel(D);

while itr<=MaxIter && (r_cdl > eprid || s_cdl > eduad)
    %%% ADMM iterations
    
    Dprv = D;
    G = G_update(Xf,Xfcnj,NX,sig,sig*fft2(D-V)+b2,A,N);
    Gr = alpha * G + (1-alpha)*D; % relaxation
    D = D_proj(Gr+V,m,H,W); % projection on constraint set
    V = Gr - D + V;

    %%
    Df = fft2(D);
    titer = toc(tsrt);
    %%
    %_________________________residuals CDL_____________________________
    nG = norm(G(:)); nD = norm(D(:)); nV = norm(V(:));
    r_cdl = norm(vec(G-D)); % primal residulal
    s_cdl = sig*(norm(vec(Dprv-D))); % dual residual
    eprid = sqrt(Nd)*eAbs+max(nD,nG)*eRel;
    eduad = sqrt(Nd)*eAbs+sig*nV*eRel;

    %_________________________sig update_____________________________
    if opts.AutoSig && rem(itr,SigUpdateCycle)==0
        [sig,V] = rho_update(sig,r_cdl,s_cdl,mu,tau,V);
    end

    %_________________________progress_______________________________
    fval = sum(vec(abs(sum(Df.*Xf,3)-Sf).^2))/(2*H*W); % residual power
    res.iterinf = [res.iterinf; [itr fval r_cdl s_cdl sig titer]];


    itr = itr+1;
end
D = D(1:m,1:m,:);
D  = D./sqrt(sum(D.^2,1:2));
end


function G = G_update(Xf,Xfcnj,NX,sig,Wf,A,N)
% G_update(Xf, abs(Xf).^2, sig, sig*fft2(D-V)+gamma*B+conj(fft2(X)).*Sf/N, A, gamma, N);
Ai = 1./(A + sig);
% Gf = Ai.*(Wf-Xf.*sum(Ai.*Xfcnj.*Wf,3)./(N+sum(NX.*Ai,3)));
Gf = Ai.*(Wf-Ai.*NX.*Wf./(N+sum(NX.*Ai,3)));
G = ifft2(Gf,'symmetric');  
end

function D = D_proj(D,m,H,W) % projection on unit ball
D = padarray(D(1:m,1:m,:,:),[H-m W-m],'post');
D  = D./max(sqrt(sum(D.^2,1:2)),1);
end


function [rho,U] = rho_update(rho,r,s,mu,tau,U)
% varying penalty parameter
a = 1;
if r > mu*s
    a = tau;
end
if s > mu*r
    a = 1/tau;
end
rho_ = a*rho;
if rho_>1e-4
    rho = rho_;
    U = U/a;
end
end



