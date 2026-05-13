% Compares regularized approximation, numerically orthogonalised 
% approximation and Vandermonde with Arnoldi.
% This script reproduces Figure 3 and 4.

% !!! This script requires Chebfun.

f = @(x) 1./(10-9*x);
m = 1000;
pts = chebpts(m);
nlist = 2:2:100;
err_backslash = []; err_lsqminnorm = []; err_vwa = []; err_QR = []; 
err_TQVD = []; err_tikh = [];
warning('off');     % ill-conditioned matrices
epsilon = 1e-14;    % regularization parameter
b = f(pts);

for n = nlist 
    n
    z = exp(1i*pi*pts/2);
    A = z.^(-n:n);

    % backslash
    c = A\b;
    err_backslash = [err_backslash norm(b - A*c,'inf')];

    % lsqminnorm
    c = lsqminnorm(A,b,epsilon);
    err_lsqminnorm = [err_lsqminnorm norm(b - A*c,'inf')];

    % Vandermonde with Arnoldi 
    [d,H] = polyfit_vwa(z,b,n);
    y = polyval_vwa(d,H,z);
    err_vwa = [err_vwa norm(b - real(y),'inf')];

    % numerical orthogonalisation via QR (using R)
    [Q,R] = qr(A,0);
    c = Q\b;
    err_QR = [err_QR norm(b - Q*c,'inf')];

    % TSVD
    [U,S,V] = svd(A); s = diag(S);
    r = sum(s >= epsilon);
    c = V(:,1:r)*((1./s(1:r)).*(U(:,1:r)'*b));
    err_TQVD = [err_TQVD norm(b - A*c,'inf')];

    % Tikhonov
    c = [A; epsilon*eye(size(A,2))] \ [b; zeros(size(A,2),1)];
    err_tikh = [err_tikh norm(b - A*c,'inf')];
end

%% plot of the convergence behaviour
figure('Position', [100 100 1000 500]); 
semilogy(nlist, err_backslash,'ok','MarkerSize',10); hold on;
semilogy(nlist, err_tikh,'.k','MarkerSize',15);
semilogy(nlist, err_TQVD,'sk','MarkerSize',10);
semilogy(nlist, err_QR,'xk','MarkerSize',10);
semilogy(nlist, err_lsqminnorm,'dk','MarkerSize',10);
semilogy(nlist, err_vwa,'+k','MarkerSize',7);
set(gca,'FontSize',18); grid on;
xlabel('n','Interpreter','latex','FontSize',22); 
ylabel('uniform error in sample points','Interpreter','latex','FontSize',22);
xticks([0 20 50 100 150]);
text(19,10^(-12.8),'based on Arnoldi basis $\rightarrow$','Interpreter','latex','FontSize',16);
legend("backslash",'backslash on Tikhonov regularized system','TSVD',...
    'numerical orthogonalisation via QR', "lsqminnorm",...
    'Vandermonde with Arnoldi','FontSize',20,'Interpreter','latex');
exportgraphics(gca, 'comparison.pdf');

%% singular value profile of the synthesis operator
Phi = @(n) chebfun(@(x) exp(1i*x*pi*(-n:n)/2));

f = figure('Position', [100 100 1200 300]);
subplot(1,3,1);
semilogy(svd(Phi(10)), '.-k', 'LineWidth', 1.5, 'MarkerSize', 20); 
ylim([1e-16 1e1]); grid on; set(gca,'FontSize',18); yticks(10.^(-15:5:0)); xticks([]);
subplot(1,3,2);
semilogy(svd(Phi(20)), '.-k', 'LineWidth', 1.5, 'MarkerSize', 20);
ylim([1e-16 1e1]); grid on; set(gca,'FontSize',18); yticks(10.^(-15:5:0)); xticks([]);
text(18/10,1e-14,'$\kappa(\mathcal{T}) \approx 1/u$','Interpreter','latex','FontSize',18);
subplot(1,3,3);
semilogy(svd(Phi(30)), '.-k', 'LineWidth', 1.5, 'MarkerSize', 20);
ylim([1e-16 1e1]); grid on; set(gca,'FontSize',18); yticks(10.^(-15:5:0)); xticks([]);
exportgraphics(f, 'comparison_svds.pdf');
