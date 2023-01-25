function [D, B, res] = CD_optimization_alg1_alt(D, S, X, A, B_old, N, opts)
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
epric = 0;
eduac = 0;

MaxIter = opts.MaxIter;
r_d= inf; s_d = inf; r_c= inf; s_c = inf;
res.iterinf = [];
mu = 5; % varying rho parameter
tau = 1.2; % varying rho parameter

vec = @(x) x(:);
itr = 1;

%% CDL CYCLES
tsrt = tic;


Xf = fft2(X);
X2 = conj(Xf).*Xf;
XS = conj(Xf).*Sf;
D = padarray(D,[H-m W-m],'post');
C = D;
U = V;
rho =sig;
Df = fft2(D);
Nd = numel(D);

while itr<=MaxIter && (r_d > eprid || s_d > eduad || r_c > epric || s_c > eduac)
    
    %%% C - ADMM iterations
    Cprv = C;
    M_1 = (X2 + rho).^(-1);
    E = ifft2( (M_1 - (M_1.^2.*X2)./(1+sum(X2.*M_1,3))).*((X2.*Df+XS)+rho*(fft2(C-U)))   ,'symmetric');
    Er = alpha * E + (1-alpha)*C; % relaxation
    C = D_proj(Er+U,m,H,W); % projection on constraint set
    U = Er - C + U;
     %%
    T = ifft2(fft2(C).*Xf,'symmetric');
    B = B_old + (1/N)*conj(Xf).*fft2(T);

    %%% D - ADMM iterations
    Dprv = D;
    G = ifft2((sig*fft2(D-V)+B)./(A+sig),'symmetric');
    Gr = alpha * G + (1-alpha)*D; % relaxation
    D = D_proj(Gr+V,m,H,W); % projection on constraint set
    V = Gr - D + V;
    %%
    Df = fft2(D);
    titer = toc(tsrt);
    %%
    %_________________________residuals CDL_____________________________
    nG = norm(G(:)); nD = norm(D(:)); nV = norm(V(:));
    r_d = norm(vec(G-D)); % primal residulal
    s_d = sig*(norm(vec(Dprv-D))); % dual residual
    eprid = sqrt(Nd)*eAbs+max(nD,nG)*eRel;
    eduad = sqrt(Nd)*eAbs+sig*nV*eRel;

    %_________________________residuals CDL_____________________________
    nE = norm(E(:)); nC = norm(C(:)); nU = norm(U(:));
    r_c = norm(vec(E-C)); % primal residulal
    s_c = rho*(norm(vec(Cprv-C))); % dual residual
    epric = sqrt(Nd)*eAbs+max(nC,nE)*eRel;
    eduac = sqrt(Nd)*eAbs+rho*nU*eRel;

    %_________________________sig update_____________________________
    if opts.AutoSig && rem(itr,SigUpdateCycle)==0
        [sig,V] = rho_update(sig,r_d,s_d,mu,tau,V);
        [rho,U] = rho_update(rho,r_c,s_c,mu,tau,U);
    end

    %_________________________progress_______________________________
    fval = sum(vec(abs(sum(Df.*Xf,3)-Sf).^2))/(2*H*W); % residual power
    res.iterinf = [res.iterinf; [itr fval r_d s_d sig 0 r_c s_c rho titer]];


    itr = itr+1;
end
D = D(1:m,1:m,:);
D  = D./sqrt(sum(D.^2,1:2));
end


function G = G_update(Xf,Xfcnj,NX,sig,Wf,A,N)
% G_update(Xf, abs(Xf).^2, sig, sig*fft2(D-V)+gamma*B+conj(fft2(X)).*Sf/N, A, gamma, N);
Ai = 1./(A + sig);
Gf = Ai.*(Wf-Xf.*sum(Ai.*Xfcnj.*Wf,3)./(N+sum(NX.*Ai,3)));
% Gf = Ai.*(Wf-Ai.*NX.*Wf./(N+sum(NX.*Ai,3)));
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
