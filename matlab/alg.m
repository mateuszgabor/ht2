function [A, B] = alg( Y, rank)
    [U, s] = left_svd_qr(Y);
    A = U(:,1:rank);
    B = A' * Y;
end