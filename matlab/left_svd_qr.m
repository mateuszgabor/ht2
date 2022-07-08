function [U, s] = left_svd_qr(A)
    if(~isfloat(A) || ndims(A) ~= 2)
      error('First argument must be a real or complex matrix.');
    end
    
    if(size(A, 1) > size(A, 2))
      [Q, R] = qr(A, 0);
      [U, S, V] = svd(R, 0);
      U = Q*U;
    else
      R = qr(A');
      R = triu(R);
      R = R(1:size(R, 2), :);
      [U, S, V] = svd(R', 0);
    end
    s = diag(S);
end