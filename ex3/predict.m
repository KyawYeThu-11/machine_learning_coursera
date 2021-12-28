function p = predict(Theta1, Theta2, X)

%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element. I
%       f your examples are in rows, then, you 
%       can use max(A, [], 2) to obtain the max for each row.
for i=1:m
    A1 = [1; X(i, :)'];
    
    Z2 = Theta1 * A1;
    A2 = [1; sigmoid(Z2)];

    Z3 = Theta2 * A2;
    A3 = sigmoid(Z3);
    
    [~, p(i)] = max(A3, [], 1);
end


end
