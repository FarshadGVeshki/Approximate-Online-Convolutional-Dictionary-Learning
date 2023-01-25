% Codes for reproducing results of the experiments on Fruit and City datasets.
% The proposed method
clear
clc

ALGORITHM = 2; % choose proposed algorithm 1 or 2

%%%% choose the dataset
dataset = 'city';
% dataset = 'fruit';


%%
K = 64;  % dictionary size
lamb = 0.1; % sparsity parameter

addpath('.\tools')
addpath('.\proposed_method')
m = 8; % filter size
rng(0)
D0 = initdict(m,K,0);
load(['.\datasets\' dataset '\train\dataset.mat'])
b =  padarray(b, [floor(m/2), floor(m/2), 0], 0, 'both');
N_train = size(b,3);
fltlmbd = 5;
for n = 1:N_train
    S = single(b(:,:,n)); S = S/255;   
    [~, S] = lowpass(S, fltlmbd);
    S_train(:,:,n) = S;
end

%% calculating lambda max using the first image in the training dataset
S = S_train(:,:,1);
t = ifft2(conj(fft2(D0,size(S,1),size(S,2))).*fft2(S),'symmetric');
lamb_max = max(abs(t(:)));
lambda = lamb*lamb_max;


%% Dictionary learning
if ALGORITHM == 1
[D, Y, rt] = func_proposed1(S_train, D0, lambda); % use for Algorithm 1
% D_ASC = D;save(['dicts\D_alg1_' dataset '_K' num2str(K) '_lamb' num2str(lamb) '.mat'],'D_ASC')
elseif ALGORITHM == 2
[D, Y, rt] = func_proposed2(S_train, D0, lambda); % use for Algorithm 2
% D_ASC = D; save(['dicts\D_alg2_' dataset '_K' num2str(K) '_lamb' num2str(lamb) '.mat'],'D_ASC')
end


Id = dict2image(D,0);
figure(1), imshow(Id,[])
%%

STr_rec = reshape(ifft2(sum(fft2(D,size(S,1),size(S,2)).*fft2(Y),3),'symmetric'), [size(S,1),size(S,2), N_train]);
RES_Tr = STr_rec - S_train;
Rtot = sum(RES_Tr(:).^2);
L1_norm = sum(abs(Y(:)));
Fval_train = (Rtot/2+ lambda*L1_norm)/N_train; % average training objective
