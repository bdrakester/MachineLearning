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


X = [ones(m,1) X];

% Loop over every training example in X
for i=1:m,
    % Compute the A2 - the hidden layer
    Xi = (X(i,:))'; 
    A2 = sigmoid(Theta1 * Xi); 
    % Add the bias unit
    A2 = [ 1; A2];
    % Compute A3, which is H, a Kx1 vector
    H = sigmoid(Theta2 * A2);
    
    % Recode the y value from number to vector yvec
    yvec = zeros(size(H,1),1); 
    yvec(y(i)) = 1;
    
    % Sum the costs for each output element
    Ji = sum( ( (-yvec) .* (log(H)) ) - ( (1 - yvec) .* (log(1-H)) ) );  
    
    % Add to the overall costs
    J = J + Ji;
endfor

J = (1/m) * J;
  
% Compute regularization
Theta1_reg = Theta1(:,2:end);
Theta2_reg = Theta2(:,2:end);

reg = (lambda / (2*m)) * ( (sum( sum( Theta1_reg.^2 ) )) + (sum( sum(Theta2_reg.^2) )) );
J = J + reg;
  

%% Part 2 - backpropagation

D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

for t = 1:m,
  % Step 1 - Perform a feedforward pass
  a1 = (X(t,:))';
  
  z2 = Theta1 * a1;
  a2 = sigmoid(z2); 
  % Add the bias unit
  a2 = [ 1; a2];
 
  z3 = Theta2 * a2; 
  a3 = sigmoid(z3);
  %fprintf('a3 = \n')
  %disp(a3)
  %fprintf('\nProgram paused. Press enter to continue.\n');
  %pause;  
  
  % Step 2 
  % First recode the y value from number to vector yvec
  yvect = zeros(size(a3,1),1); 
  yvect(y(t)) = 1;
  
  delta3 = a3 - yvect;
  
  % Step 3 
  delta2 = ((Theta2(:,2:end))' * delta3) .* sigmoidGradient(z2);
  
  % Step 4
  D1 = D1 + (delta2 * a1');
  D2 = D2 + (delta3 * a2');
  
endfor

Theta1_grad = (1/m) .* D1;
Theta2_grad = (1/m) .* D2;

Theta1_grad_reg = (lambda / m) .* Theta1(:,2:end);
Theta2_grad_reg = (lambda / m) .* Theta2(:,2:end);

Theta1_grad_reg = [zeros(size(Theta1_grad_reg,1),1) Theta1_grad_reg];
Theta2_grad_reg = [zeros(size(Theta2_grad_reg,1),1) Theta2_grad_reg];

Theta1_grad = Theta1_grad + Theta1_grad_reg;
Theta2_grad = Theta2_grad + Theta2_grad_reg;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
