function [x, info] = SAGA_lstsq_minibach(A, b, parameter)
    % Parameter setting
    [n, d]       = size(A);                          
    epoch_max    = parameter.epoch_max;           % Max epoch number
    gamma        = parameter.gamma;               % Learning rate
    x            = parameter.x0;                  % Initial condition
    lambda       = parameter.lambda;              % Regularization
    s            = parameter.s;                   % minibatch size
    % Output initialization    
    info         = struct('iter_time',[],'fx',[],'epoch',[]);
    % Initialization
    g_phi = zeros(n,d);
    for i = 1 : n
        g_phi(i,:) = A(i,:)* (A(i,:)*x' - b(i));
    end
    g_phi_av = mean(g_phi,1);
    for epoch = 1 : epoch_max       
        for j = 1 : n
            tic
            perm = randperm(n);
            i = perm(1:s);
            % Update the next iteration
            gx         = n * A(i,:) .* repmat((A(i,:)*x' - b(i)),[1,d]);
            w_next     = x - gamma * (mean(gx,1) - mean(g_phi(i,:)) + g_phi_av);
            x_next     = 1 / (1+lambda*gamma) * w_next;
            g_phi(i,:) = gx;     
            g_phi_av   = mean(g_phi,1);
            % Save information
            info.iter_time(end+1) = toc;
            info.fx(end+1) = 0.5 * norm(A*x'-b,2)^2;               
            disp([0.5 * (norm(A*x_next'-b,2)^2)])
            % Prepare the next iteration
            x = x_next;            
        end
    end        
    info.epoch = epoch;
end