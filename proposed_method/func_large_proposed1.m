function [rt] = func_large_proposed1(dataset, H, W, Ntr, D, lamb)
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


    [X, ~] = CSC(D,  S, lambda);

    A = (1-1/n)*A + (1/n)*abs(fft2(X)).^2;
    [D, B, ~] = CD_optimization_alg1(D, S, X, A, (1-1/n)*B, n);
    disp(['iteration ' num2str(n)])

    if any(n == [1 10 100 1000])
        D_ASC = D;save(['dicts\D_alg1_' dataset '_K' num2str(K) '_lamb' num2str(lamb) '_iter' num2str(n) '.mat'],'D_ASC')

    end
rt(n) = toc(tt);
end


end