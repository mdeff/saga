%% Exam of stochastic gradient descent

%% Set the problem dimensions and the noise level.
n  = 1e4;
p  = 1e2;
s  = 1e1;
noiseSNRdB = 30;


%% Generate a s-sparse vector with unit norm.
x_org = zeros(p, 1);
index = randperm(p, s);
x_org(index) = randn(s, 1);
Radius = 1;
x_org = x_org/norm(x_org,1)*Radius;

%% Generate measurement matrix
A     = normr(randn(n, p));

%% Evaluate the Lipschitz constant.
b       = awgn(A*x_org, noiseSNRdB, 'measured'); 

%% Define the gradient/function.

cvx_begin 
    cvx_precision best
    variable x_cvx(p)
    minimize( .5*sum_square(A*x_cvx - b) )
    subject to 
        norm(x_cvx,1) <= Radius;
cvx_end


%%
mu = eigs(A'*A, 1, 'SA');

param.fMin = cvx_optval;
param.xMin = x_cvx;
param.maxit = 1e5;
param.tolx = 1e-6;
param.c = mu/2;

% [PGA.x, PGA.info]               = PGA(fx, gx, gradf, x0, Lips, lambda, maxit, tolx, fmin, x_org);
[~, info.SG] = SG(A, b, Radius, param);
param2 = param;
param2.c = param.c/info.SG.M_2;
%[~, info.ASG] = ASG(A, b, Radius, param2);
% [LS_PGA.x, LS_PGA.info]         = LS_PGA(fx, gx, gradf, x0, Lips, lambda, maxit, tolx, fmin, x_org);
% [LS_FPGA.x, LS_FPGA.info]       = LS_FPGA(fx, gx, gradf, x0, Lips, lambda, maxit, tolx, fmin, x_org);
% [LS_FPGA_R.x, LS_FPGA_R.info]   = LS_FPGA_R(fx, gx, gradf, x0, Lips, lambda, maxit, tolx, fmin, x_org);
% [FPGA_R.x, FPGA_R.info]         = FPGA_R(fx, gx, gradf, x0, Lips, lambda, maxit, tolx, fmin, x_org);

%% 
M_2 = info.SG.M_2; % Not exact, but approximated to some accuracy
c = param.c;
Lips = norm(A'*A);
ThBoundObj = 1./(1:param.maxit)' * (Lips/2*max(c^2*M_2/(2*c*mu - 1), norm(x_cvx,2)^2 ));
ThBoundDist = 1./(1:param.maxit)' * max(c^2*M_2/(2*c*mu - 1), norm(x_cvx,2)^2 );
%%
close all
figure( 'Position', [100 100 1000 400])
% subplot(221)
% semilogy(ThBoundDist, '--', 'LineWidth', 2);
% hold on
% xlabel('iterations', 'FontSize', 16);
% ylabel('$\| \mathbf{x}^k - \mathbf{x}^\star \|^2$', 'Interpreter', 'latex', 'FontSize',18)
% xlim([0,param.maxit])
% semilogy(info.SG.distx, 'LineWidth', 2);
% %semilogy(info.ASG.distx);
% set(gca,'FontSize',14)
% ax = legend('Theoretical bound', 'Empirical performance');
% set(ax, 'FontSize' , 16)

subplot(121)
loglog((1:param.maxit)/n,ThBoundDist, '--', 'LineWidth', 2);
hold on
loglog((1:length(info.SG.distx))/n,info.SG.distx, 'LineWidth', 2);
xlabel('epoch', 'FontSize', 16);
ylabel('$\| \mathbf{x}^k - \mathbf{x}^\star \|^2$', 'Interpreter', 'latex', 'FontSize',18)
xlim([.01,param.maxit/n])
set(gca,'XTick',10.^(-10:10))
set(gca,'FontSize',14)
ax = legend('Theoretical bound', 'Empirical performance','Location','southwest');
set(ax, 'FontSize' , 16 )

%loglog(info.ASG.distx);


% subplot(222)
% semilogy(ThBoundObj, '--', 'LineWidth', 2);
% hold on
% semilogy(info.SG.fx- cvx_optval, 'LineWidth', 2);
% xlabel('iterations', 'FontSize', 16);
% ylabel('$f(\mathbf{x}^k) - f(\mathbf{x}^\star)$', 'Interpreter', 'latex', 'FontSize',18)
% xlim([0,param.maxit])
% set(gca,'FontSize',14)
% %semilogy(info.ASG.fx- cvx_optval);


subplot(122)
loglog((1:param.maxit)/n,ThBoundObj, '--', 'LineWidth', 2);
hold on
loglog((1:length(info.SG.distx))/n,info.SG.fx- cvx_optval, 'LineWidth', 2);
xlabel('epoch', 'FontSize', 16);
ylabel('$f(\mathbf{x}^k) - f(\mathbf{x}^\star)$', 'Interpreter', 'latex', 'FontSize',18)
xlim([.01,param.maxit/n])
set(gca,'FontSize',14)
set(gca,'XTick',10.^(-10:10))

%loglog(info.ASG.fx- cvx_optval);
