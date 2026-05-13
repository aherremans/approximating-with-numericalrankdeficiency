function [d,H] = polyfit_vwa(x,f,n)
      m = length(x);
      Q = ones(m,1);
      H = zeros(n+1,n);
      for k = 1:n
          q = x.*Q(:,k);
          for j = 1:k
              H(j,k) = Q(:,j)'*q/m;
              q = q - H(j,k)*Q(:,j);
          end
          H(k+1,k) = norm(q)/sqrt(m);
          Q = [Q q/H(k+1,k)];
      end 
d = [real(Q) imag(Q(:,2:n+1))]\f;
d = d(1:n+1) - 1i*[0; d(n+2:2*n+1)];