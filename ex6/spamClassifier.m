function spamClassifier(model)
% Given a model as the argument and a file name as the input, it classifies
% the file as spam or not.

cont = 'y';

while (strcmp(cont, 'y'))
    prompt = 'Enter the file name that you want to classify as spam or not:';
    filename = input(prompt,'s');
    file_contents = readFile(filename);
    word_indices  = processEmail(file_contents);
    features = emailFeatures(word_indices);
    
    if svmPredict(model, features) == 0
        disp('Hooray! This email is not a spam.😄');
    else
        disp('Yikes! This email is suspicious.🙄');
    end

    prompt = 'Do you want to continue using our classifier? Y/n';
    cont = lower(input(prompt, 's'));
    
    if strcmp(cont, 'n')
        break;
    end
    disp("=========================");
end