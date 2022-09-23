function [ myRNN ] = initialize_rnn( pred_par, beh_par, p, M)
% Initialization of the variables controlling the internal dynamics of the RNN :
%   - the synaptic weights  
%   - the system states x
%   - selection of the activation function
%   - ...
% The variable p represents the dimension of the RNN output space.
% The symnaptic weights are  randomly distributed according to a normal distribution of standard deviation sg = pred_par.Winit_std_dev. 
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    m = pred_par.SHL*p;
    q = pred_par.rnn_state_space_dim;
    sg = pred_par.Winit_std_dev;

    myRNN.input_space_dim = m;
    myRNN.state_space_dim = q;
    myRNN.output_space_dim = p;
    myRNN.nb_weights = q*(p+q+m+1);
    
    % evaluation variables initialization
    myRNN.Ypred = zeros(p, M);
    myRNN.pred_loss_function = zeros(M, 1, 'single');
    myRNN.pred_time_array = zeros(M, 1, 'single');  
    
    % weights initialization
    myRNN.Wa = normrnd(0, sg, [q, q]);
    myRNN.Wb = normrnd(0, sg, [q, m+1]); % "+1" because of the bias unit
    myRNN.Wc = normrnd(0, sg, [p, q]);

    % states initialization
    myRNN.x = zeros(q, 1);

    % activation function and its derivative
    myRNN.phi = @(v) tanh(v);
    myRNN.phi_prime = @(v) 1./((cosh(v)).^2);
    
    %% Initialization of the variables specific to the training method chosen
    
    switch(pred_par.pred_meth_idx)
        case 2 % RNN RTRL
            
            % state space dynamics 3D tensor  
            myRNN.LBDA = zeros(q, q + m + 1, q);
                % myRNN.LBDA(:,:,j) is the matrix LBDA_{j,n} , ie the Jacobian matrix of x_n as a function of w_j (j in 1,...q)
                % cf the Haykin's book

            % w(:,j) corresponds to the w_j matrix at time n in Haykin's book
            % w_j = [wa_j , wb_j]
            % with Wa^T = [wa_1, ..., wa_q] and Wb^T = [wb_1, ..., wb_q]
            myRNN.w = zeros(m+q+1, q);

            % gradient of the instantaneous energy loss En with respect to each entry in w (corresponding to the gradients with respect to Wa and Wb) - cf Haykin's book
            myRNN.w_gradient = zeros(m+q+1, q);

            % matrix U{:,:,j) ( "U_{j,n}" in Haykin's book)
            myRNN.U = zeros(q, m+q+1, q);    
            
        case 5 % RNN UORO

            % Variables xtilde and theta_tilde such that 
            % dx/dtheta is approximated by xtilde*theta_tilde

            myRNN.xtilde = zeros(myRNN.state_space_dim, 1);
                % named stilde in the paper from Tallec et al.
            myRNN.theta_tilde = zeros(1, myRNN.nb_weights);
            
            myRNN.dtheta = zeros(1, myRNN.nb_weights);
            myRNN.dtheta_g = zeros(1, myRNN.nb_weights);
            
    end

    if (pred_par.update_meth == 2) % ADAM
         switch(pred_par.pred_meth_idx)
            case 2 % RNN RTRL
                myRNN.grad_moments.m_t = zeros(m+p+q+1, q);
                myRNN.grad_moments.v_t = zeros(m+p+q+1, q);
            case 5 % RNN UORO
                myRNN.grad_moments.m_t = zeros(1, myRNN.nb_weights);
                myRNN.grad_moments.v_t = zeros(1, myRNN.nb_weights);           
         end  
    else
        myRNN.grad_moments = struct();
    end    
    
    if beh_par.GPU_COMPUTING
        
        myRNN.Ypred = gpuArray(myRNN.Ypred);
        myRNN.pred_loss_function = gpuArray(myRNN.pred_loss_function);
        myRNN.pred_time_array = gpuArray(myRNN.pred_time_array);

        myRNN.Wa = gpuArray(myRNN.Wa);
        myRNN.Wb = gpuArray(myRNN.Wb);
        myRNN.Wc = gpuArray(myRNN.Wc);
        myRNN.x = gpuArray(myRNN.x);
        
        switch(pred_par.pred_meth_idx)
            case 2 % RNN RTRL
                myRNN.LBDA = gpuArray(myRNN.LBDA);
                myRNN.w = gpuArray(myRNN.w);
                myRNN.w_gradient = gpuArray(myRNN.w_gradient);
                myRNN.U = gpuArray(myRNN.U);
            case 5 % RNN UORO
                myRNN.xtilde = gpuArray(myRNN.xtilde);
                myRNN.theta_tilde = gpuArray(myRNN.theta_tilde);
                myRNN.dtheta = gpuArray(myRNN.dtheta);
                myRNN.dtheta_g = gpuArray(myRNN.dtheta_g);
        end
        
        if (pred_par.update_meth == 2) % ADAM
            myRNN.grad_moments.m_t = gpuArray(myRNN.grad_moments.m_t);
            myRNN.grad_moments.v_t = gpuArray(myRNN.grad_moments.v_t);                 
        end        
        
    end

        
end