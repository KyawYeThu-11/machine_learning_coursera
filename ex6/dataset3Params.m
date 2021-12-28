function [C, sigma] = dataset3Params(X, y, Xval, yval)
% It returns the optimal choice of C and sigma based on a cross-validation set
% for Part 3 of the exercise

values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. 

initial_err = Inf;

for i=1:length(values)
    for j=1:length(values)
        C_var = values(i);
        sigma_var = values(j);

        model = svmTrain(X, y, C_var, @(x1, x2)gaussianKernel(x1, x2, sigma_var));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        
        if err < initial_err
            initial_err = err;
            C = C_var;
            sigma = sigma_var;
        end
        
    end
end






% =========================================================================

end
