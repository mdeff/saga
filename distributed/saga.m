function [x, info] = saga(A, b, fx, gradf, prox, parameter)
    % Parameter setting
    [n, d]       = size(A);                          
    epoch_max    = parameter.epoch_max;           % Max epoch number
    gamma        = parameter.gamma;               % Learning rate
    tol          = parameter.tol;                 % convergence tolerance
    x            = parameter.x0;                  % Initial condition
    
    % Output initialization    
    info         = struct('iter_time',[],'fx',[],'epoch',[]);
    % Initialization
    g_phi = zeros(n,d);
    for i = 1 : n
        g_phi(i,:) = gradf(x,i);
    end
    g_phi_av = mean(g_phi,1);
    flag     = true;
    for epoch = 1 : epoch_max        
        if ~flag
            break
        end
        for j = 1 : n
            tic
            if epoch == 0
                i = j;
            else
                i = randi(n);
            end
            % Update the next iteration
            gx         = gradf(x,i);
            w_next     = x - gamma * (gx - g_phi(i,:) + g_phi_av);
            g_phi_av   = g_phi_av - g_phi(i,:) / n + gx / n;
            g_phi(i,:) = gx;          
            x_next     = prox(w_next, gamma);
            % Save information
            info.iter_time(end+1) = toc;
            info.fx = fx(x);   
            print('Suboptimality is ' + str(abs(fx(x)-fx(x_nature)) ))
            % Check optimality condition
            if abs(fx(x_next)-fx(x)) <= tol
                flag = false;
                break
            end
            % Prepare the next iteration
            x = x_next;
        end
    end        
    info.epoch = epoch;
end