function [new_theta] = update_param_optim(theta, dtheta, optim_par, grad_moments, t)
% Optimization function
% theta and dtheta need to have the same dimensions, this is the only requirement
%
% Author : Pohl Michel
% Date : April 11th, 2021
% Version : v2.0
% License : 3-clause BSD License
 
    eta = optim_par.learn_rate;

    if optim_par.GRAD_CLIPPING
        thresh = optim_par.grad_threshold;
        grad_norm = sqrt(sum(dtheta.^2, 'all'));
        if (grad_norm > thresh)
            dtheta = (thresh/grad_norm)*dtheta;
        end        
    end
    
    switch(optim_par.update_meth)
        case 1 % gradient descent
            new_theta = theta - eta*dtheta;
            
        case 2 % ADAM (adaptive moment estimation)
            
            beta1 = optim_par.ADAM_beta1;
            beta2 = optim_par.ADAM_beta2;
            
            grad_moments.m_t = beta1*grad_moments.m_t + (1-beta1)*dtheta;
            grad_moments.v_t = beta2*grad_moments.v_t + (1-beta2)*(dtheta.^2);
            
            m_t_bias_corrected = (grad_moments.m_t)/(1-beta1^t);
            v_t_bias_corrected = (grad_moments.v_t)/(1-beta2^t);
            
            new_theta = theta - eta*m_t_bias_corrected./(optim_par.ADAM_eps + sqrt(v_t_bias_corrected));
                % use of broadcasting
            
    end


end

