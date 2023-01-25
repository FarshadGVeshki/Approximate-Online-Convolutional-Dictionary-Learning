% Codes for reproducing results of the experiments on Fruit-large dataset.
% The proposed methods 

%%
clear
clc

%%
%% chose algorithm 1 or 2
ALGORITHM = 2; % choose proposed algorithm 1 or 2

%%
dataset = 'flickr_large';
K = 100;  % dictionary size
lamb = 0.1; % sparsity parameter
Ntr = 1000;
H = 256; % image size 
%%
addpath('.\tools')
addpath('.\proposed_method')
m = 8; % filter size
rng(0)
D0 = initdict(m,K,0);


%% Dictionary learning

if ALGORITHM == 1
    rt = func_large_proposed1(dataset, H, H, Ntr, D0, lamb);
elseif ALGORITHM == 2
    rt = func_large_proposed2(dataset, H, H, Ntr, D0, lamb);
end

