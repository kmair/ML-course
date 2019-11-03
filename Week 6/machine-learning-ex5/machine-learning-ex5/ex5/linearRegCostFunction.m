function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%x0 = zeros([m,1])
%x = [x0 X]
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%% Loss
R = length(theta);  % R = No. of terms
%disp(length(theta))
for i = 2:R
    J = J + lambda/(2*m) * theta(i)^2;
end

h = X*theta;

for j = 1:m
    J = J + 1/(2*m) * (h(j) - y(j))^2;
end
disp(J)
%% Gradient

for i = 1:m
    grad(1) = grad(1) + 1/m * (h(i) - y(i));
    
    for j = 2:R
        grad(j) = grad(j) + 1/m * (h(i) - y(i)) * X(i,j);
    end
end
for j = 2:R
    grad(j) = grad(j) + lambda/m * theta(j);
end        
        
% =========================================================================

grad = grad(:);
end
