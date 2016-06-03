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
b_tr = b_tr - mean(b_tr);
b_te = b_te - mean(b_tr);
mu = min(eig(A_tr'*A_tr/n));
L  = norm(A_tr'*A_tr/n);
param.epoch_max = 20;
param.gamma = 1/(3*(mu*n+1));
param.x0 = zeros(1, d);
param.lambda = 5e-4;
param.m = 8;
param.s = 50;

%% Experiment 1 
param1 = param;
param1.gamma = 1/(3*(mu*n+1));
[~, info1] = SAGA_lstsq_dist(A_tr, b_tr, param1);
param2 = param;
param2.gamma = 2/(3*(mu*n+1));
[~, info2] = SAGA_lstsq_dist(A_tr, b_tr, param2);
param3 = param;
param3.gamma = 5e-1/(3*(mu*n+1));
[~, info3] = SAGA_lstsq_dist(A_tr, b_tr, param3);
param4 = param;
param4.gamma = 1e-1/(3*(mu*n+1));
[~, info4] = SAGA_lstsq_dist(A_tr, b_tr, param4);
param5 = param;
param5.gamma = 5e-2/(3*(mu*n+1));
[~, info5] = SAGA_lstsq_dist(A_tr, b_tr, param5);
figure;
semilogy(info1.fx,'linewidth',2); hold on
semilogy(info2.fx,'linewidth',2);
semilogy(info3.fx,'linewidth',2);
semilogy(info4.fx,'linewidth',2);
semilogy(info5.fx,'linewidth',2);
xlim([1, param.epoch_max])
xlabel('# of epoch')
ylabel('f(x)')
legend('\gamma = \gamma_0', '\gamma = 2 \gamma_0', '\gamma = 0.5 * \gamma_0', ...
       '\gamma = 0.1 * \gamma_0', '\gamma = 0.05 * \gamma_0')
%% Experminet 2
param1 = param;
param1.epoch_max = 1; 
param1.gamma;
[~, info1] = SAGA_lstsq_vec(A_tr, b_tr, param1);
[~, info2] = SAGA_lstsq_parfor(A_tr, b_tr, param1);
[~, info3] = SAGA_lstsq_for(A_tr, b_tr, param1);

figure;
semilogy(cumsum(info1.iter_time),'linewidth',2); hold on
semilogy(cumsum(info2.iter_time),'linewidth',2);
semilogy(cumsum(info3.iter_time),'linewidth',2);

xlabel('# of iteration')
ylabel('time(s)')
legend('for loop', 'parfor loop', 'vectorize')
%% Experiment 3
param1 = param;
param1.epoch_max = 5; 
param1.gamma = 1/(3*(mu*n+1));
[~, info1] = SAGA_lstsq_vec(A_tr, b_tr, param1);
param2 = param;
param2.epoch_max = 5;
param2.gamma = 2/(3*(mu*n+1));
[~, info2] = SAGA_lstsq_vec(A_tr, b_tr, param2);
param3 = param;
param3.epoch_max = 5;
param3.gamma = 5e-1/(3*(mu*n+1));
[~, info3] = SAGA_lstsq_vec(A_tr, b_tr, param3);
param4 = param;
param4.epoch_max = 5;
param4.gamma = 1e-1/(3*(mu*n+1));
[~, info4] = SAGA_lstsq_vec(A_tr, b_tr, param4);
param5 = param;
param5.epoch_max = 5;
param5.gamma = 5e-2/(3*(mu*n+1));
[~, info5] = SAGA_lstsq_vec(A_tr, b_tr, param5);
%%
figure;
semilogy(info1.fx,'linewidth',2); hold on
semilogy(info2.fx,'linewidth',2);
semilogy(info3.fx,'linewidth',2);
semilogy(info4.fx,'linewidth',2);
semilogy(info5.fx,'linewidth',2);
xlabel('# of iteration')
ylabel('f(x)')
legend('\gamma = \gamma_0', '\gamma = 2 \gamma_0', '\gamma = 0.5 * \gamma_0', ...
       '\gamma = 0.1 * \gamma_0', '\gamma = 0.05 * \gamma_0')
%% Experiment 4
param1 = param;
param1.gamma = 0.1/(3*(mu*n+1));
[x1, info1] = SAGA_lstsq(A_tr, b_tr, param1);
param2 = param;
param2.gamma = 1/(3*(mu*n+1));
[~, info2] = SAGA_lstsq_dist(A_tr, b_tr, param2);
fx2 = repmat(info2.fx,[n,1]);
param3 = param;
param3.gamma = 1/(3*(mu*n+1));
[~, info3] = SAGA_lstsq_vec(A_tr, b_tr, param3);
%%
figure;
semilogy(info1.fx,'linewidth',2); hold on
semilogy(fx2(:),'linewidth',2); 
semilogy(info3.fx,'linewidth',2); 
xlabel('# of iterations')
ylabel('f(x)')
legend('SAGA','Distributed SAGA I','Distributed SAGA II');

figure;
fx1 = reshape(info1.fx,n,[]); fx1 = [info1.fx(1) fx1(end,:)];
t1 = reshape(info1.iter_time,n,[]); t1 = [0 sum(t1,1)];
semilogy(cumsum(t1),fx1,'linewidth',2); hold on
semilogy([0 cumsum(info2.iter_time)], [info1.fx(1) info2.fx],'linewidth',2);
fx3 = reshape(info1.fx,n,[]); fx3 = [info1.fx(1) fx3(end,:)];
t3 = reshape(info3.iter_time,n,[]); t3 = [0 sum(t3,1)];
semilogy(fx3,'linewidth',2); 
xlabel('time(s)')
ylabel('f(x)')
legend('SAGA','Distributed SAGA I','Distributed SAGA II');
xlim([0 90])