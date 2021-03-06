function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
for row=1:size(g, 1)
    for column=1:size(g, 2)
        g(row, column) = 1/(1+(exp(-z(row, column))));
    end



% =============================================================

end
