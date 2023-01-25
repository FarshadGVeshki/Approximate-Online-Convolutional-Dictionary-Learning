clear
clc

%% choose proposed algorithm

% alg = 'alg1'; % algorithm 1
alg = 'alg2'; % algorithm 2

addpath('.\tools')
fltlmbd = 5;
Nte = 10;
dataset = 'flickr_large';
lamb = 0.1; % lambda = 0.1*lambda_max (calculated for the first image in training dataset)
H = 256;
K=100;

%% lambda max
rng(0)
D0 = initdict(8,K,0);
S = single(rgb2gray(imread(['.\datasets\' dataset '\' num2str(9999+1) '.jpg'])))/255; %%% first image in the dataset
[h, w] = size(S);
m = min(h,w);
S = S( floor((h-m)/2)+1 :end - ceil((h-m)/2) , floor((w-m)/2)+1 :end - ceil((w-m)/2));
S = imresize(S,[H H]);
[~, S] = lowpass(S, fltlmbd);

t = ifft2(conj(fft2(D0,size(S,1),size(S,2))).*fft2(S),'symmetric');
lamb_max = max(abs(t(:)));
lambda = lamb*lamb_max;
%%


c = 1;
for iii = [1 10 100 1000]
load(['.\dicts\D_' alg '_flickr_large_K' num2str(K) '_lamb0.1_iter' num2str(iii) '.mat'])
% load(['.\dicts\D_' alg '_flickr_large_K' num2str(K) '_lamb0.1_iter' num2str(iii) '.mat'])

D = D_ASC;

for n = 1:Nte
    S = single(rgb2gray(imread(['.\datasets\' dataset '_test\' num2str(11000+n) '.jpg'])))/255; %%% modify for each dataset
    [h, w] = size(S);
    m = min(h,w);
    S = S( floor((h-m)/2)+1 :end - ceil((h-m)/2) , floor((w-m)/2)+1 :end - ceil((w-m)/2));
    S = imresize(S,[H H]);   
    [~, S] = lowpass(S, fltlmbd);
    STe(:,:,n) = S;
end

[Y, ~] = CSC(D,  STe, lambda);
STe_rec = reshape(ifft2(sum(fft2(D,size(S,1),size(S,2)).*fft2(Y),3),'symmetric'), [size(S,1),size(S,2), Nte]);
RES_Te = STe_rec - STe;
Rtot = sum(RES_Te(:).^2);
Sparsitytot = sum(abs(Y(:)));
Fval_Te = (Rtot/2+ lambda*Sparsitytot)/Nte;
respower = Rtot/Nte;
PSNR_Te = psnr(STe_rec(:),STe(:));

Obj(c) = Fval_Te; % objective values
c = c+1;
end
