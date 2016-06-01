function [x, info] = SAGA_lstsq(A, b, parameter, x_org)
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
        g_phi(i,:) = A(i,:)* (A(i,:)*x' - b(i));
    end
    g_phi_av = mean(g_phi,1);
    flag     = true;
    for epoch = 1 : epoch_max        
        if ~flag
            break
        end
        for j = 1 : n
            tic
            if epoch == 1
                i = j;
            else
                i = randi(n);
            end
            % Update the next iteration
            gx         = A(i,:)* (A(i,:)*x' - b(i));
            x_next     = x - gamma * (gx - g_phi(i,:) + g_phi_av);
%             g_phi_av   = g_phi_av - g_phi(i,:) / n + gx / n;
            g_phi(i,:) = gx;     
            g_phi_av   = mean(g_phi,1);
            % Save information
            info.iter_time(end+1) = toc;
            info.fx = norm(A*x'-b,2)^2;   
            % Check optimality condition
            if abs(norm(A*x_org'-b,2)^2 - norm(A*x'-b,2)^2) <= tol
                flag = false;
                break
            else
                disp([j (norm(A*x_next'-b,2)^2)])
            end
            % Prepare the next iteration
            x = x_next;
        end
    end        
    info.epoch = epoch;
end