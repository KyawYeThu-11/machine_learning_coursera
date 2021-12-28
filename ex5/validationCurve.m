function [best_lambda, lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%  It returns the train and validation errors (in error_train, error_val)
%  for different values of lambda. You are given the training set (X,
%  y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
best_lambda = 0;


% Number of training examples
m = size(X, 1);
error_lambda = 0;

% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
for i = 1:length(lambda_vec)
    theta = trainLinearReg(X, y, lambda_vec(i));
    [error_train(i), ~] = linearRegCostFunction(X, y, theta, error_lambda);
    % 1/(2*X_m) * sum((X * theta - y).^2);

    [error_val(i), ~] = linearRegCostFunction(Xval, yval, theta, error_lambda);
    % 1/(2*Xval_m) * sum((Xval * theta - yval).^2);
end

error_combo = error_train + error_val;
[~, index] = min(error_combo);
best_lambda = lambda_vec(index);

end
