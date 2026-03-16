% Compares regularized approximation, numerically orthogonalised 
% approximation and Vandermonde with Arnoldi.
% This script reproduces Figure 3, 4 and 6(a).

% !!! This script requires Chebfun.

addpath("src")
f = @(x) 1./(10-9*x);
pts = chebpts(1000);
nlist = 2:2:100;
err1 = []; err2 = []; err3 = []; err4 = []; err5 = [];
condR = [];
warning('off'); % ill-conditioned matrices

for n = nlist 
    n
    z = exp(1i*pi*pts/2);
    
    % Matlab backslash = regularized approximation
    c = polyfit(z,f(pts),n);
    y = polyval(c,z);
    err1 = [err1 norm(f(pts) - real(y),'inf')];

    % Vandermonde with Arnoldi 
    [d,H] = polyfitA(z,f(pts),n);
    y = polyvalA(d,H,z);
    err2 = [err2 norm(f(pts) - real(y),'inf')];

    % numerical orthogonalisation via QR (using R)
    [c,R] = polyfitQR(z,f(pts),n);
    y = polyvalQR(c,R,z);
    err3 = [err3 norm(f(pts) - real(y),'inf')];
    condR = [condR skeel(R)];

    % numerical orthogonalisation via QR (using Q)
    A = z.^(0:n);
    [Q,R] = qr(A,0);
    d = [real(Q) imag(Q(:,2:n+1))]\f(pts);
    d = d(1:n+1) - 1i*[0; d(n+2:2*n+1)];
    err4 = [err4 norm(f(pts) - real(Q*d),'inf')];

    % truncated SVD approximation
    A = z.^(0:n);
    A = [real(A) imag(A(:,2:n+1))];
    [U,S,V] = svd(A);
    r = sum(diag(S) >= 5e-14);
    c = V(:,1:r)*(S(1:r,1:r)\(U(:,1:r)'*f(pts)));
    err5 = [err5 norm(f(pts) - real(A*c),'inf')];
end

%% plot of the convergence behaviour
figure2('Position', [100 100 800 500]); 
semilogy(nlist, err1,'ok','MarkerSize',10); hold on;
semilogy(nlist, err2,'.k','MarkerSize',20);
semilogy(nlist, err3,'+k','MarkerSize',7);
semilogy(nlist, err4,'xk','MarkerSize',10);
semilogy(nlist, err5,'sk','MarkerSize',10);
set(gca,'FontSize',18); grid on;
xlabel('n','Interpreter','latex','FontSize',22); 
ylabel('uniform error in sample points','Interpreter','latex','FontSize',22);
xticks([0 20 50 100 150]);
legend("Matlab's backslash",'Vandermonde with Arnoldi', ...
    'numerical orthogonalisation (using R)', ...
    'numerical orthogonalisation (using Q)', ...
    'TSVD approximation','FontSize',20);

figure2('Position', [100 100 400 500]);
semilogy(nlist,condR,'.-k','MarkerSize',20,'LineWidth',2);
set(gca,'FontSize',18); grid on; ylabel('Skeel condition number of R','Interpreter','latex','FontSize',22);
xlabel('n','Interpreter','latex','FontSize',22);

%% singular value profile of the synthesis operator
frame = @(n) chebfun(@(x) real(exp(1i*x*pi*(0:n)/2)));

f = figure2('Position', [100 100 1200 300]);
subplot(1,3,1);
semilogy(svd(frame(10)), '.-k', 'LineWidth', 1.5, 'MarkerSize', 20); 
ylim([1e-16 1e1]); grid on; set(gca,'FontSize',18); yticks(10.^(-15:5:0)); xticks([]);
text(11/10,1e-14,'$A_n > \epsilon_\textit{mach}^2 B_n$','Interpreter','latex','FontSize',18);
subplot(1,3,2);
semilogy(svd(frame(20)), '.-k', 'LineWidth', 1.5, 'MarkerSize', 20);
ylim([1e-16 1e1]); grid on; set(gca,'FontSize',18); yticks(10.^(-15:5:0)); xticks([]);
text(18/10,1e-14,'$A_n \approx \epsilon_\textit{mach}^2 B_n$','Interpreter','latex','FontSize',18);
subplot(1,3,3);
semilogy(svd(frame(30)), '.-k', 'LineWidth', 1.5, 'MarkerSize', 20);
ylim([1e-16 1e1]); grid on; set(gca,'FontSize',18); yticks(10.^(-15:5:0)); xticks([]);
text(30/10,1e-14,'$A_n < \epsilon_\textit{mach}^2 B_n$','Interpreter','latex','FontSize',18);
