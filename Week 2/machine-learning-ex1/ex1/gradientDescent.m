function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);  %stores values of J on every iteration

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    %fprintf('Cost function value at iteration %d: %f\n', iter, J_history(iter));
    temp1 = X*theta;
    temp2 = temp1 - y;
    temp3 = X' * temp2;
    temp4 = alpha/m;
    temp3 = temp4*temp3;
    temp5 = theta - temp3;
    theta = temp5;
    % ============================================================
    J_history(iter) = computeCost(X, y, theta);
    % Save the cost J in every iteration    
    %J_history(iter) = computeCost(X, y, theta);

endfor

end
