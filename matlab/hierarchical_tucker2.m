function [ Xtt, Xp ] = hierarchical_tukcer2( X, ranks)

    ranks = cell2mat([ranks{:}]);
    Xtt = cell(size(ranks,1),1);
    dim = size(X);
    N = ndims(X);
    X_ = X;
    
    W = reshape(permute(double(X),[1 [1:0,2:N]]),dim(1), prod(dim)/dim(1));
    Xtt{1} = alg(W, ranks(1));
    X_ = contraction(X_, Xtt{1}, 1, 1);
    
    W = reshape(permute(double(X),[3 [1:2,4:N]]),dim(3), prod(dim)/dim(3));
    Xtt{2} = alg(W, ranks(2));
    X_ = contraction(X_, Xtt{2}, 2, 1);
    
    Xp = X_;
    
    X = permute(X_,[3 1 4 2]);
    S = reshape(double(X), ranks(1) * size(X,2), ranks(2)*size(X,4));
    S1 = alg(S, ranks(3));
    Xtt{3} = reshape(S1,[ranks(1) size(X,2) ranks(3)]);
    X_ = S1' * S;
    Xtt{4} = permute(reshape(X_, ranks(3), ranks(2), size(X,4)),[2 3 1]);
    
end
    
    