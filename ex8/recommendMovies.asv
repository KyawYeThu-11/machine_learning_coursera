function recommendMovies(num_top)
%RECOMMENDMOVIES recommends movies for you 

% Load data (Y and R will be loaded)
% Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 943 users
% R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
load('ex8_movies.mat');
 
% Load movie list
movieList = loadMovieList();

% Initialize my ratings
my_ratings = zeros(1682, 1);

while 1
    movie_id = input('Enter the ID of the movie you want to rate:');
    formatSpec = "How much rating you want to give to '%s'? (1-5)\n";
    prompt = compose(formatSpec,movieList{movie_id});
    rating = input(prompt);

    my_ratings(movie_id) = rating;

    cont = input('Do you wish to continue rating movies? Y/n','s');
    disp('---------------------------');
    if upper(cont) == 'Y'
        continue
    else
        break
    end
end

% print the user's ratings
fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
    end
end

%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj','on','MaxIter',100);

% Set Regularization
lambda = 10;

disp('Predicting top recommendations for you...');

theta = fmincg(@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, num_features,lambda)), initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), num_users, num_features);

% predict ratings of all movies by each user
p = X * Theta';
% We can add mean value to every rating but since we're dealing only with
% our prediction, we do so only for that 
my_predictions = p(:,1) + Ymean;

[r, ix] = sort(my_predictions,'descend');
for i=1:10
    j = ix(i);
    if i == 1
        fprintf('\nTop recommendations for you:\n');
    end
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), movieList{j});
end

for i = 1:length(my_ratings)
    if i == 1
        fprintf('\n\nOriginal ratings provided:\n');
    end
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
    end
end

end