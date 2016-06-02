function [x, info] = SAGA_lstsq_par(A, b, parameter)
    % Parameter setting
    [n, d]       = size(A);                          
    epoch_max    = parameter.epoch_max;           % Max epoch number
    gamma        = parameter.gamma;               % Learning rate
    x            = parameter.x0;                  % Initial condition
    lambda       = parameter.lambda;              % Regularization
    m            = parameter.m;                   % # of cpu cores
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
            x_sub = zeros(m,d);
            ind    = zeros(m,1);
            gx_sub = zeros(m,d);
            for k = 1 : m
                i = randi(n);
                % Update the next iteration
                gx          = A(i,:)* (A(i,:)*x' - b(i));
                w_next      = x - gamma * (gx - g_phi(i,:) + g_phi_av);
                x_sub(k,:)  = 1 / (1+lambda*gamma) * w_next;
                ind(k)      = i;
                gx_sub(k,:) = gx;
                
            end   
            g_phi(ind,:) = gx_sub;
            g_phi_av = mean(g_phi,1);
            x_next = mean(x_sub);
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