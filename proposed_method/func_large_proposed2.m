function [rt] = func_large_proposed2(dataset, H, W, Ntr, D, lamb)
K = size(D,3);
A = 0;
B = 0;
fltlmbd = 5;
tt = tic;
for n = 1:Ntr
    S = single(rgb2gray(imread(['.\datasets\' dataset '\' num2str(9999+n) '.jpg'])))/255; %%% modify for each dataset
    [h, w] = size(S);
    m = min(h,w);
    S = S( floor((h-m)/2)+1 :end - ceil((h-m)/2) , floor((w-m)/2)+1 :end - ceil((w-m)/2));
    S = imresize(S,[H W]);   
    [~, S] = lowpass(S, fltlmbd);

    if n == 1
        t = ifft2(conj(fft2(D,size(S,1),size(S,2))).*fft2(S),'symmetric');
        lamb_max = max(abs(t(:)));
        lambda = lamb*lamb_max;
    end    

    [X, ~] = CSC(D,  S, lamb);
    [D, ~] = D_optimization_alg2(D, S, X, A, B, n);

    Xf = fft2(X);
    [Tf, ~, ~] = C_optimization_alg2(D, Xf, S);
    A = (1/(n+1))*sum(abs(Xf).^2,4) + (n/(n+1))*A;
    B = (1/(n+1))*sum(Tf.*conj(Xf),4) + (n/(n+1))*B;
    disp(['iteration ' num2str(n)])

    if any(n == [1 10 100 1000])
        D_ASC = D;save(['dicts\D_alg2_' dataset '_K' num2str(K) '_lamb' num2str(lamb) '_iter' num2str(n) '.mat'],'D_ASC')
    end
rt(n) = toc(tt);

end


end