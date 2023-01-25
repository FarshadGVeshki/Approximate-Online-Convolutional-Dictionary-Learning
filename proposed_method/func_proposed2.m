function [D,Y,rt] = func_proposed2(STr, D, lamb)
A = 0;
B = 0;
Ntr = size(STr,3);

tt = tic;
for n = 1:Ntr
    S = STr(:,:,n);

    [X, ~] = CSC(D,  S, lamb); % convolutional sparse approximation
    Y(:,:,:,n) = X;

    [D, ~] = D_optimization_alg2(D, S, X, A, B, n); % optimize D

    Xf = fft2(X);
    [Tf, ~, ~] = C_optimization_alg2(D, Xf, S); % optimize C
    A = (1/(n+1))*sum(abs(Xf).^2,4) + (n/(n+1))*A; % update Alpha and Beta (history arrays)
    B = (1/(n+1))*sum(Tf.*conj(Xf),4) + (n/(n+1))*B;
    disp(['iteration ' num2str(n)])

end
rt = toc(tt);

end