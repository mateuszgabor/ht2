function [res] = ht2_conv_decomposition(X, R1, R3, R13)
    ranks = {{R1,R3},{R13}};
    
    [res, ~] = hierarchical_tucker2(X, ranks);    
    Gs = contraction(res{3},res{4},3,3); 
    Xs = permute(contraction(contraction(Gs,res{1},1,2),res{2},2,2),[3 1 4 2]); 
    R = X - Xs;
    r = norm(R(:))/norm(X(:));
    disp("Residual error");
    disp(r)
    
end
    