function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

for i=1:m
    for K=1:num_labels
        a1 = [1; X(i,:)'];
    
        z2 = Theta1*a1;
        a2 = [1; sigmoid(z2)];
    
        z3 = Theta2*a2;
        a3 = sigmoid(z3);
        % Now you have unoptimised hypothesis of each unit in output layer
        % You have to calculate the cost of each hypothesis by comparing each hypothesis with 
        % actual result, which is in the form of one and zero
        temp_y = (y(i) == K);
        J = J+(-(temp_y'*log(a3(K)))-((1-temp_y)'*log(1-a3(K))));
    end
end

% regularisation
regular_theta1 = 0;
for j=1:hidden_layer_size
    for k=2:input_layer_size+1

        regular_theta1 = regular_theta1 + (Theta1(j, k))^2;
    end
end

regular_theta2 = 0;
for j=1:num_labels
    for k=2:hidden_layer_size+1
        regular_theta2 = regular_theta2 + (Theta2(j, k))^2;
    end
end

J = (1/m) * J + (lambda/(2*m))*(regular_theta1+regular_theta2);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

delta3 = zeros(num_labels, 1);
accumulator1 = Theta1_grad;
accumulator2 = Theta2_grad;

for t=1:m
    a1 = [1; X(t,:)'];
   
    z2 = Theta1*a1;
    a2 = [1; sigmoid(z2)];

    z3 = Theta2*a2;
    a3 = sigmoid(z3);

    for k=1:num_labels
        delta3(k) = a3(k) - (y(t) == k);
    end

    delta2 = Theta2'*delta3 .* sigmoidGradient([1; z2]);
    
    accumulator1 = accumulator1 + delta2(2:end)*a1';
    accumulator2 = accumulator2 + delta3*a2';
end

Theta1_grad = accumulator1/m;
Theta2_grad = accumulator2/m;

% Part 3: Implement regularization with the cost function and gradients.

Theta1_grad_regular = zeros(size(Theta1));
Theta2_grad_regular = zeros(size(Theta2));

for i=1:hidden_layer_size
    for j=2:input_layer_size+1
        Theta1_grad_regular(i, j) = (lambda/m)*(Theta1(i, j));
    end
end

for i=1:num_labels
    for j=2:hidden_layer_size+1
        Theta2_grad_regular(i, j) = (lambda/m)*(Theta2(i, j));
    end
end

Theta1_grad = Theta1_grad + Theta1_grad_regular;
Theta2_grad = Theta2_grad + Theta2_grad_regular;

%-----------------------------------------------
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
