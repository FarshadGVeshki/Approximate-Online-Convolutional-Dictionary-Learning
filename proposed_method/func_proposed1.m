function [D,Y,rt] = func_proposed1(STr, D, lamb)
A = 0;
B = 0;
Ntr = size(STr,3);

tt = tic;
for n = 1:Ntr
    S = STr(:,:,n);

    [X, ~] = CSC(D,  S, lamb); % convolutional sparse approximation
    Y(:,:,:,n) = X; 
    A = (1-1/n)*A + (1/n)*abs(fft2(X)).^2; % update history array alpha 
    [D, B, ~] = CD_optimization_alg1(D, S, X, A, (1-1/n)*B, n); % optimize the dictionary and update history array beta
    disp(['iteration ' num2str(n)])

end
rt = toc(tt);

end