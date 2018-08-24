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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% ----------Finding the cost function----------

h = X * theta;
theta(1) = 0;
r_term = (lambda/(2*m)) * (sum(theta .^ 2));

J = [(1 / (2*m)) * sum((h - y) .^ 2)] + r_term;

% ----------------Finding grad----------------

errors_vector = h - y;
grad = [(X' * errors_vector) .* (1/m)] + [(lambda/m) * theta];

% =========================================================================

grad = grad(:);

end
