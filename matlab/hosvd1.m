function [U, G] = hosvd1(X, R)
    n=1;
    dim = size(X);
    N = ndims(X);
    
    W = reshape(permute(double(X),[n [1:n-1,n+1:N]]),dim(n), prod(dim)/dim(n));
    [U, Gm] = alg(W, R);
    G = reshape(Gm, [R dim(1:n-1) dim(n+1:end)]);
    
end
    
    