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


%-----------------------Calculating Cost Function----------------------------
z = X * theta;
error = z - y;
J = (1 / (2 * m)) * (error' * error);

reg_theta = theta(2:end);
reg_theta = reg_theta .^ 2;
reg_theta = (lambda / (2 * m)) * sum(sum(reg_theta));
J = J + reg_theta;

%-----------------------Calculating Gradients for Theta----------------------
grad = (1 / m) * (X' * error);
reg_gradient = theta(2:end);  %ignoring the bias term theta0
reg_gradient = [0; reg_gradient]; %adding 0 for the bias term gradient
grad = grad + (lambda / m) * reg_gradient;
% =========================================================================

grad = grad(:);

end
