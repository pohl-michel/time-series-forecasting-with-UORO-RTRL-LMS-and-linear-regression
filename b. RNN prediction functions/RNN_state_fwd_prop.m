function [z, new_state_x] = RNN_state_fwd_prop(myRNN, u, x)
% input u
% output x
% returns the new state x'
% and the neural input to the activation function (used when doing backprop)
% This function does not change the value of myRNN.x
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License

    z = myRNN.Wa*x + myRNN.Wb*u;
    new_state_x = myRNN.phi(z);

end

