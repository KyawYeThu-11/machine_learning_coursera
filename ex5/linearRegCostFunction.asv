function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

% The function computes the cost of using theta as the parameter for 
% linear regression to fit the data points in X and y. Returns the cost 
% in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% We don't want to regularize theta0 term, which is represented as theta(1)
% since indexing in MATLAB starts from 1
temp = theta;
temp(1) = 0;

% If X hasn't been added a column of ones -> X = [ones(m, 1) X];

% calculating J
J = 1/(2*m) * sum((X * theta - y).^2) + (lambda/(2*m))*sum(temp.^2);

% calculating gradient
grad = 1/m * (X' * ((X * theta) - y)) + lambda/m * temp; 
grad = grad(:);

end
