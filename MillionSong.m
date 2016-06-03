clc
clearvars
try
   load('data.mat')
catch
    A = load('YearPredictionMSD.txt');
    b = A(:,1);
    A(:,1) = [];
end
A = normr(A);
d = size(A,2);
n = 1e4; %463715
A_tr = A(1:n,:);
A_te = A(n+1:end,1);
b_tr = b(1:n);
b_te = b(n+1:end);
mu = min(eig(A_tr'*A_tr/n));
L  = norm(A_tr'*A_tr/n);
param.epoch_max = 2;
param.gamma = 1/(3*(mu*n+1));
param.x0 = zeros(1, d);
param.lambda = 1e-3;
param.m = 10;
param.s = 50;
%% SAGA
param1 = param;
param1.gamma = 5e-3/(3*(mu*n+1));
[x1, info1] = SAGA_lstsq(A_tr, b_tr, param1);
%% SAGA-Distributed I
param2 = param;
param2.gamma = 1e-1/(3*(mu*n+1));
[x2, info2] = SAGA_lstsq_dist(A_tr, b_tr, param2);
%% SAGA-Distributed II
param3 = param;
param3.gamma = 1e-3/(3*(mu*n+1));
[x3, info3] = SAGA_lstsq_par(A_tr, b_tr, param2);
%% SAGA-MiniBach
param4 = param;
param4.gamma = 5e-1/(3*(mu*n+1));
[x4, info4] = SAGA_lstsq_minibach(A_tr, b_tr, param4);
%% plot
figure;
semilogy(info1.fx,'linewidth',2); hold on
semilogy(info2.fx,'linewidth',2,'linestyle','-.');
semilogy(info3.fx,'linewidth',2,'linestyle','--');
semilogy(info4.fx,'linewidth',2,'linestyle',':');
xlabel('# of iterations')
ylabel('cost value')
legend('SAGA','Distributed SAGA I','Distributed SAGA II','Mini Bach SAGA');