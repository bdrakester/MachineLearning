function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i = 1:size(X,1)

  
  % Compute distances between current example and each centroid.
  % Store in distances vector, where distances(j) is distance from centroid j
  
  % For loop solution
  %x_i = (X(i,:))'; % Create Vector of current example
  %distances = zeros(K,1);
  %for j = 1:size(distances,1)
  %  mu_j = centroids(j,:)';
  %  distances(j) = sum( (x_i - mu_j).^2 ); 
  %endfor
  
  % Vecotorized solution
  x_i = X(i,:);
  D = bsxfun(@minus, x_i, centroids);
  distances = sum(D.^2,2);
  
  
  % Find index of smallest distance, ie. closest centroid. Store in idx(i),
  % indicating this is the centroid closest to example i.
  [min_val, min_index] = min(distances);
  idx(i) = min_index; 
  
 
  
endfor



% =============================================================
end

