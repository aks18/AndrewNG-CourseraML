function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
theta_size = size(theta, 1);
%h = g(z), where g = sigmoid function & z = X * theta
z = X * theta;  
h = sigmoid(z); 
%calculating log(h) & log(1-h)
temp1 = log(h);
temp2 = log(1-h);
%calculating y'*log(h) & (1-y)'*log(1-h)
temp3 = y' * temp1;
temp4 = (1 - y)' * temp2;
%calculating y'*log(h) + (1-y)'*log(1-h)
temp5 = temp3 + temp4;
temp6 = -(1/m);
%calculating cost = -(1/m)*(y'*log(h) + (1-y)'*log(1-h))
J = temp6 * temp5;

%Calculating Regularization Term [lambda/2m(sum(theta^2))]
%ignoring the bias term from theta, as that should not be regularized
temp_theta = theta(2:end); 
theta_square = temp_theta'*temp_theta;
temp10 = lambda/(2*m);
temp11 = temp10 * theta_square;

J = J + temp11;

%Calculating Gradient = (1/m)*(X' * (h - y))
temp7 = h - y;
temp8 = X' * temp7;
temp9 = (1/m);
grad = temp9 * temp8;

%Calculating the lambda/m(theta)
temp12 = lambda/m;
temp13 = temp12*temp_theta;
grad_theta = [0;temp13];
%Adding gradient to calculated gradient vector
grad = grad+grad_theta;
% =============================================================

grad = grad(:);

end
