function [Zf, DD, res] = C_optimization_alg2(D, Xf, S, opts)
%% implementation of the step for optimization of {c_k} in Algorithm 2
Sf = fft2(S);
[H,W,P] = size(Sf);
[m,~,K] = size(D);
Sf = reshape(Sf,[H W 1 P]);

if nargin < 4
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
    opts.Vinit = zeros(H,W,K,P,'single');
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
V = opts.Vinit;
sig = opts.sig;
SigUpdateCycle = opts.SigUpdateCycle;

eAbs = opts.eAbs;
eRel = opts.eRel;
epri = 0;
edua = 0;

MaxIter = opts.MaxIter;

r= inf; s = inf;
res.iterinf = [];
mu = 5; % varying rho parameter
tau = 1.2; % varying rho parameter

vec = @(x) x(:);
itr = 1;

%% CDL CYCLES
tsrt = tic;

D = padarray(D,[H-m W-m],'post');
Nd = numel(D);
while itr<=MaxIter && (r > epri || s > edua)
    Dprv = D;
    G = G_update(Xf,Sf,sig,fft2(D-V));
    Gr = alpha * G + (1-alpha)*D; % relaxation
    D = D_proj(Gr+V,m,H,W); % projection on constraint set
    if opts.dcfilter == 1
        D(:,:,1) = Dprv(:,:,1);
    end
    V = Gr - D + V;

    %%
    Df = fft2(D);
    titer = toc(tsrt);
    %%

    %_________________________residuals CDL_____________________________
    nG = norm(G(:)); nD = norm(D(:)); nV = norm(V(:));
    r = norm(vec(G-D)); % primal residulal
    s = sig*norm(vec(Dprv-D)); % dual residual
    epri = sqrt(Nd)*eAbs+max(nG,nD)*eRel;
    edua = sqrt(Nd)*eAbs+sig*nV*eRel;
    

    %_________________________sig update_____________________________
    if opts.AutoSig && rem(itr,SigUpdateCycle)==0
        [sig,V] = rho_update(sig,r,s,mu,tau,V);
    end

    %_________________________progress_______________________________
    fval = sum(vec(abs(sum(Df.*Xf,3)-Sf).^2))/(2*H*W); % residual power
    res.iterinf = [res.iterinf; [itr fval r s sig titer]];

    itr = itr+1;
end
DD = D(1:m,1:m,:);
DD  = DD./sqrt(sum(DD.^2,1:2));
Zf = Df.*Xf;
Z = ifft2(Zf,'symmetric');
Zf = fft2(Z);
end


function G = G_update(Xf,Sf,sig,Wf)
C = conj(Xf)./(sum(abs(Xf).^2,3)+sig);
Rf = Sf - sum(Xf.*Wf,3); % residual update
Gf = Wf + C.*Rf;
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