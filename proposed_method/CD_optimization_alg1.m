function [D, B, res] = CD_optimization_alg1(D, S, X, A, B_old, N, opts)
%% implementation of the optimization algorithm in Algorithm 1

[H,W] = size(S);
[m,~,K] = size(D);

if nargin < 9
    opts = [];
end
if ~isfield(opts,'MaxIter')
    opts.MaxIter = 300;
end
if ~isfield(opts,'sig')
    opts.rho = 10;
end
if ~isfield(opts,'AutoSig')
    opts.AutoRho= 1;
end
if ~isfield(opts,'SigUpdateCycle')
    opts.RhoUpdateCycle = 1;
end
if ~isfield(opts,'dcfilter')
    opts.dcfilter = 0;
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
V = zeros(H,W,K,'single');
U = zeros(H,W,K,'single');;
rho = opts.rho;
RhoUpdateCycle = opts.RhoUpdateCycle;


eAbs = opts.eAbs;
eRel = opts.eRel;
epri = 0;
edua = 0;


MaxIter = opts.MaxIter;
r_cd= inf; s_cd = inf;
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
Ndc = numel(D)*2;
C = D;
G = D;
E = D;
while itr<=MaxIter && (r_cd > epri || s_cd > edua)
    
    %%% C - ADMM iterations
    Cprv = C;
    Dprv = D;

    a = (X2 + N*rho).^(-1);
    E = ifft2( (a - (a.^2.*X2)./(1+sum(X2.*a,3))).*((X2.*fft2(G)+XS)+N*rho*(fft2(C-U)))   ,'symmetric');
    B = B_old + (1/N)*conj(Xf).*fft2(ifft2(fft2(E).*Xf,'symmetric'));
    G = ifft2((rho*fft2(D-V)+B)./(A+rho),'symmetric');

    Er = alpha * E + (1-alpha)*C; % relaxation
    Gr = alpha * G + (1-alpha)*D; % relaxation

    C = D_proj(Er+U,m,H,W); % projection on constraint set
    D = D_proj(Gr+V,m,H,W); % projection on constraint set
    
    U = Er - C + U;
    V = Gr - D + V;
     %%
%     T = ifft2(fft2(C).*Xf,'symmetric');
%     B = B_old + (1/N)*conj(Xf).*fft2(T);

    
    %%
    Df = fft2(D);
    titer = toc(tsrt);
    %%
    %_________________________residuals CDL_____________________________
    nEG = norm([G(:);E(:)]); nCD = norm([D(:);C(:)]); nUV = norm([V(:); U(:)]);
    r_cd = norm([vec(G-D); vec(C-E)]); % primal residulal
    s_cd = rho*(norm([vec(Dprv-D);vec(Cprv-C)])); % dual residual
    epri = sqrt(Ndc)*eAbs+max(nCD,nEG)*eRel;
    edua = sqrt(Ndc)*eAbs+rho*nUV*eRel;

    %_________________________sig update_____________________________
    if opts.AutoRho && rem(itr,RhoUpdateCycle)==0
        UV(:,:,:,1) = U;
        UV(:,:,:,2) = V;
        [rho,UV] = rho_update(rho,r_cd,s_cd,mu,tau,UV);
        U = UV(:,:,:,1);
        V = UV(:,:,:,2);
    end

    %_________________________progress_______________________________
    fval = sum(vec(abs(sum(Df.*Xf,3)-Sf).^2))/(2*H*W); % residual power
    res.iterinf = [res.iterinf; [itr fval r_cd s_cd rho titer]];


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
