clc
clearvars
n = 1e3;
p = 5e2;
A = normr(randn(n, p));
x_org = randn(p, 1);
x_org = x_org/norm(x_org,2);
b = awgn(A*x_org, 30, 'measured');
mu = eigs(A'*A, 1, 'SA');
L  = norm(A'*A);
parameter.epoch_max =20;
parameter.gamma = 0.2; %1/(2*(mu*n+L));
parameter.tol = 1e-4;
parameter.x0 = zeros(1, p);
% parameter.x0 = parameter.x0/norm(parameter.x0,2);
[x, info] = SAGA_lstsq(A, b, parameter, x_org');
