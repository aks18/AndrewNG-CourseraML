function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%Part 1 Solution: Feed Forward Cost Calculation in a Neural Network

X = [ones(m, 1) X]; %X(5000x401)
for i = 1:m
  %Input Layer
  x_i = X(i,:); %taking every example stored in rows, one by one, x_i(1x401)
  x_i = x_i'; %converting the row vector into a column vector, x_i(401x1)
  layer1_output = Theta1 * x_i; %Theta1(25x401) * x_i(401x1) = layer1_output(25x1)
  layer1_output = sigmoid(layer1_output);
  %1st Hidden Layer
  layer2_input = [1; layer1_output]; %layer2_input(26x1)
  layer2_output = Theta2 * layer2_input; %Theta2(10x26) * layer2_input (26x1) = layer2_output(10x1)
  layer2_output = sigmoid(layer2_output); %layer2_output(10x1)
  h_i = layer2_output;
  %Output Layer
  y_i = zeros(num_labels, 1);
  pres_label = y(i);
  y_i(pres_label) = 1;
  y_i = y_i';
  cost_i = (y_i * log(h_i)) + ((1-y_i) * log(1 - h_i));
  J = J + cost_i;
endfor
J = (-1/m) * J;

%Calculating Regularization Term
Theta1_temp = Theta1(: , 2:end); %removing the bias term's weights
Theta2_temp = Theta2(: , 2:end);
Theta1_square = Theta1_temp .^ 2; %element-wise square
Theta2_square = Theta2_temp .^ 2; 
Theta1_sum = sum(Theta1_square); %taking sum of the columns
Theta1_sum = sum(Theta1_sum); %taking sum of th resulting vector
Theta2_sum = sum(Theta2_square);
Theta2_sum = sum(Theta2_sum);
regularization_sum = Theta1_sum + Theta2_sum;
regularization_sum = (lambda/(2*m)) * regularization_sum;

J = J + regularization_sum;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
