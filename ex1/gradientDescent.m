function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
num_theta = length(theta);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    theta_copy = theta;
    for feature = 1:num_theta
        sum = 0;
        for example = 1:m
            sum = sum + ((theta'*X(example,:)')-y(example))*X(example,feature);
        end
        theta_copy(feature) = theta(feature)-((sum/m)*alpha);
    end
    theta = theta_copy;
    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);
end

end
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %