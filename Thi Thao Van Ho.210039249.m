
% ****************************
%  IF2211 coursework
%  Lecturer: Chuanping Sun
%  Bayes Business School
%  City University of London
% ****************************

clear all; clc; warning off;


%% readtable / treat missing values / specify import types
opts = detectImportOptions('trainData.csv');
opts.MissingRule = 'omitrow'; % missing values can cause problematic estimation results
                              % some functions are robust with missing data,
                              % while some are not

categoricalVariableNames = {'grade', 'emp_length', 'home_ownership',  ...
       'collections_12_mths_ex_med', 'application_type', 'acc_now_delinq'};
opts = setvartype(opts, categoricalVariableNames, 'categorical' );
                    
opts = setvaropts(opts, 'emp_length', 'TreatAsMissing', {'n/a'});

% remove variables that have too many missing values
miss_var = {'mths_since_last_delinq'}; % this variable has 110000+ missing data
selected = setdiff(opts.VariableNames(3:end), miss_var);
opts.SelectedVariableNames = selected;

% readtable with options
trainTbl = readtable('trainData.csv', opts);

%convert loan_status to logical values
key = 'Charged Off';
trainTbl.loan_status = strcmpi(trainTbl.loan_status, key);

summary(trainTbl)

% check variable levels to determine which variables are categorical 
func = @(x) numel(unique(x));
varLevel = varfun(func,trainTbl); 

% -------------------------------------------------------------------------
% similarly for testdata

testTbl = readtable('testData.csv', opts);

%convert loan_status to logical values
testTbl.loan_status = strcmpi(testTbl.loan_status, key);

summary(testTbl)
% -------------------------------------------------------------------------
% group predictor variables
predictorNames = setdiff(trainTbl.Properties.VariableNames, 'loan_status');
numVariableNames = setdiff(predictorNames, categoricalVariableNames); % numerical predictors

%% convert categorical variables to dummy variables 
% because some functions such as corr(), ridge() and lassoglm() cannot deal with categorical predictors

% create dummy variable for 'grade'
grade_dum = dummyvar(trainTbl.grade);
grade_dum2 = dummyvar(testTbl.grade);
types = categories(trainTbl.grade);
dum_names = join([repmat({'grade'}, length(types), 1),types], '_' ); 
grade_trainTbl = array2table(grade_dum, 'VariableNames', dum_names);
grade_testTbl = array2table(grade_dum2, 'VariableNames', dum_names);

% create dummy variables for 'emp_length'
emp_length_dum = dummyvar(trainTbl.emp_length);
emp_length_dum2 = dummyvar(testTbl.emp_length);
types = categories(trainTbl.emp_length);
dum_names = join ([repmat({'emp_length'}, length(types), 1),types], '_' );  
emp_length_trainTbl = array2table(emp_length_dum, 'VariableNames', dum_names);
emp_length_testTbl = array2table(emp_length_dum2, 'VariableNames', dum_names);

% create dummy variable for 'home_ownership'
home_ownership_dum = dummyvar(trainTbl.home_ownership);
home_ownership_dum2 = dummyvar(testTbl.home_ownership);
types = categories(trainTbl.home_ownership);
dum_names = join([repmat({'home_ownership'}, length(types), 1),types], '_' ); 
home_ownership_trainTbl = array2table(home_ownership_dum, 'VariableNames', dum_names);
home_ownership_testTbl = array2table(home_ownership_dum2, 'VariableNames', dum_names);

% create dummy variable for 'collections_12_mths_ex_med'
collections_12_mths_ex_med_dum = dummyvar(trainTbl.collections_12_mths_ex_med);
collections_12_mths_ex_med_dum2 = dummyvar(testTbl.collections_12_mths_ex_med);
types = categories(trainTbl.collections_12_mths_ex_med);
dum_names = join([repmat({'collections_12_mths_ex_med'}, length(types), 1),types], '_' ); 
collections_12_mths_ex_med_trainTbl = array2table(collections_12_mths_ex_med_dum, 'VariableNames', dum_names);
collections_12_mths_ex_med_testTbl = array2table(collections_12_mths_ex_med_dum2, 'VariableNames', dum_names);

% create dummy variable for 'application_type'
application_type_dum = dummyvar(trainTbl.application_type);
application_type_dum2 = dummyvar(testTbl.application_type);
types = categories(trainTbl.application_type);
dum_names = join([repmat({'application_type'}, length(types), 1),types], '_' ); 
application_type_trainTbl = array2table(application_type_dum, 'VariableNames', dum_names);
application_type_testTbl = array2table(application_type_dum2, 'VariableNames', dum_names);

% create dummy variable for 'acc_now_delinq'
acc_now_delinq_dum = dummyvar(trainTbl.acc_now_delinq);
acc_now_delinq_dum2 = dummyvar(testTbl.acc_now_delinq);
types = categories(trainTbl.acc_now_delinq);
dum_names = join([repmat({'acc_now_delinq'}, length(types), 1),types], '_' ); 
acc_now_delinq_trainTbl = array2table(acc_now_delinq_dum, 'VariableNames', dum_names);
acc_now_delinq_testTbl = array2table(acc_now_delinq_dum2, 'VariableNames', dum_names);


% create predictor table. Note that first column of dummy variables are
% used as reference level

Predictor_trainTbl = [trainTbl(:,numVariableNames), grade_trainTbl(:,2:end), ...
               emp_length_trainTbl(:,2:end), home_ownership_trainTbl(:,2:end), ...
               collections_12_mths_ex_med_trainTbl(:,2:end), ...
               application_type_trainTbl(:,2:end), acc_now_delinq_trainTbl(:,2:end) ];
           
Predictor_testTbl = [testTbl(:,numVariableNames), grade_testTbl(:,2:end), ...
               emp_length_testTbl(:,2:end), home_ownership_testTbl(:,2:end), ...
               collections_12_mths_ex_med_testTbl(:,2:end), ...
               application_type_testTbl(:,2:end), acc_now_delinq_testTbl(:,2:end) ];

new_trainTbl = [Predictor_trainTbl, trainTbl(:,{'loan_status'})];
new_testTbl = [Predictor_testTbl, testTbl(:,{'loan_status'})];


%% preliminary analysis / covariance analysis
% In this block, using the new_trainTbl to analyse the correlation between
% predicors and the response variable 'loan_status'. Use a heatmap to
% display the correlation structure. And find out the top 10 and bottom 10
% correlated predictors, call them top10 and bottom10. Include the graph in
% your reporting. 
% #########################################################################
% -------------------------------------------------------------------------

%move loan_status to the first column
new_trainTbl = movevars(new_trainTbl,'loan_status','Before',1);
%compute correlation coefficient
corr_coeff = corr(new_trainTbl{:, :}, new_trainTbl{:,'loan_status'});
vnames = new_trainTbl.Properties.VariableNames(:,:);
%draw correlation coefficient colormap
figure
imagesc(corr_coeff)
colorbar;
set(gca,'XTick',1:length(vnames), 'XTickLabel', vnames,'XTickLabelRotation',45,  ...
         'YTick', 1:length(vnames), 'YTickLabel', vnames, 'TickLabelInterpreter', 'none');
title('correlation-coefficent-colormap')

%sort correlation as descendence 
[sortedCorr, idx] = sort(corr_coeff, 'descend');
%find out top 10 and bottom 10 correlated predictors
top10 = new_trainTbl.Properties.VariableNames(idx(1:10));
bottom10 = new_trainTbl.Properties.VariableNames(idx(end-9:end-1));

% -------------------------------------------------------------------------
% #########################################################################

%% GLS 
% estimate a logistical model using the trainTbl (i.e., contains categorical variables)
% hint: use the function fitglm().
% #########################################################################
% -------------------------------------------------------------------------

%Determine which variables are catagories

varLevel

%since we have been given the categorical variables we input that GLS

%Logistical model

 

mdl1 = fitglm(trainTbl, 'ResponseVar', 'loan_status', 'CategoricalVar', categoricalVariableNames, 'Distribution', 'Binomial');

% -------------------------------------------------------------------------
% #########################################################################

% forecast using training data, and compute glsMSE_train
% #########################################################################
% -------------------------------------------------------------------------


y_train_predict = predict(mdl1, trainTbl);
% compute glsMSE_train (mean square error)
glsMSE_train = mean((trainTbl.loan_status - y_train_predict).^2);
glsMSE_train
% -------------------------------------------------------------------------
% #########################################################################

% forecast using test data, and compute glsMSE_test
% #########################################################################
% -------------------------------------------------------------------------
yhat_test = predict(mdl1, testTbl);

% Set threshold value of 0.5
yhat_test_binary = yhat_test >= 0.5;

% Calculate the mean squared error (MSE) between predicted and actual values
glsMSE_test = mean((testTbl.loan_status - yhat_test_binary).^2);
glsMSE_test

% -------------------------------------------------------------------------
% #########################################################################



%% stepwise regression models
% Fit a stepwise regression model using only the top10 predictors you have
% found in the covariance analysis above. 
swMdl = stepwiseglm(new_trainTbl(:, [top10; {'loan_status'}]),  'constant', 'upper', 'linear', ...
              'Distribution', 'binomial' );
          
new_trainTbl1 = new_trainTbl(:, [top10, {'loan_status'}]);
swMdl = stepwiseglm(new_trainTbl1, 'constant', 'Upper', 'linear', 'Distribution', 'binomial');

% forecast using training sample, and compute MSE for the training sample
swForecast_train = predict(swMdl, Predictor_trainTbl(:,top10));
swForecast_train = (swForecast_train >=0.5); 
swMSE_train = mean((swForecast_train - trainTbl.loan_status).^2);
fprintf('MSE of the step-wise estimator using training sample is: %6.4f \n', swMSE_train);

%true
Predictor_trainTbl1 = new_trainTbl(:, top10);
swForecast_train = predict(swMdl, Predictor_trainTbl1);
swForecast_train = (swForecast_train >=0.5); 
swMSE_train = mean((swForecast_train - trainTbl.loan_status).^2);
fprintf('MSE of the step-wise estimator using training sample is: %6.4f \n', swMSE_train);

% forecast using testing sample, and compute MSE for the testing sample, 
% call it swMSE_test.
% #########################################################################
% -------------------------------------------------------------------------

Predictor_test = new_testTbl(:, top10);
swForecast_test = predict(swMdl, Predictor_test);
swForecast_test = (swForecast_test >=0.5); 
swMSE_test = mean((swForecast_test - testTbl.loan_status).^2);
fprintf('MSE of the step-wise estimator using testing sample is: %6.4f \n', swMSE_test);


% -------------------------------------------------------------------------
% #########################################################################


     
                      
%% lasso
% model estimation using training data
% lasso cannot handle categorical predictors
[B_lasso,fitInfo] = lassoglm(Predictor_trainTbl{:,:}, trainTbl.loan_status, 'binomial', 'CV', 3);

B0 = fitInfo.Intercept(fitInfo.IndexMinDeviance);
coef = [B0; B_lasso(:, fitInfo.IndexMinDeviance)];

% predict loan_status using training sample
lassohat_train = glmval(coef, Predictor_trainTbl{:,:}, 'logit'); % specify 'logit' for binomial response 
lassoForecast_train = (lassohat_train >=0.5);
lassoMSE_train = mean((lassoForecast_train -trainTbl.loan_status).^2);
fprintf('MSE of the lasso estimator using training sample is: %6.4f \n', lassoMSE_train); 
% output confusion chart for training sample
figure()
c_lasso_train = confusionchart(trainTbl.loan_status, lassoForecast_train);
title('LASSO confusion chart using training data')

% predict loan_status using testing sample. Hint: follow the method we used
% for training sample
% #########################################################################
% -------------------------------------------------------------------------

%model estimation using testing data
% lasso cannot handle categorical predictors      
[B_lasso2,fitInfo2] = lassoglm(Predictor_testTbl{:,:}, testTbl.loan_status, 'binomial', 'CV', 3);
B0_test = fitInfo2.Intercept(fitInfo2.IndexMinDeviance);
coef_test = [B0_test; B_lasso2(:, fitInfo2.IndexMinDeviance)];
%predict loan_Status using test sample
lassohat_test = glmval(coef_test, Predictor_testTbl{:,:}, 'logit'); % specify 'logit' for binomial response 
lassoForecast_test = (lassohat_test >=0.5);
lassoMSE_test = mean((lassoForecast_test -testTbl.loan_status).^2);
fprintf('MSE of the lasso estimator using testing sample is: %6.4f \n', lassoMSE_test); 


% -------------------------------------------------------------------------
% #########################################################################
% output confusion chart for testing sample
figure()
c_lasso_test = confusionchart(testTbl.loan_status, lassoForecast_test);
title('LASSO confusion chart using testing data')

% plot the lasso coefficient (B_lasso)along lambda values, and save the figure 
% as png file for reporting. You don't need to standarise the dataset.
figure()
% #########################################################################
% -------------------------------------------------------------------------
% plot lasso coefficient using lamda
lassoPlot(B_lasso,fitInfo,'plottype','lambda','xscale','log');
%label x 
xlabel('Lambda');
%label y
ylabel(' Lasso Coefficients');
title('Lasso Coefficients along Lambda values');
%save figure
saveas(gcf, 'lasso_coefficients.png');


% -------------------------------------------------------------------------
% #########################################################################


%% Elastic Net
% Do the same analysis as you did with the lasso, but use the elastic
% net model with alpha parameter set to 0.5.
% #########################################################################
% -------------------------------------------------------------------------

% model estimation using training sample
% elastic net cannot handle categorical predictors
[B_enet, fitInfo] = lassoglm(Predictor_trainTbl{:,:}, trainTbl.loan_status, 'binomial', 'CV', 3, 'Alpha', 0.5);
B0 = fitInfo.Intercept(fitInfo.IndexMinDeviance);
coef_enet = [B0; B_enet(:, fitInfo.IndexMinDeviance)];

%predict loan_status using train data
enetResponse_train = glmval(coef_enet, Predictor_trainTbl{:,:}, 'logit'); % specify 'logit' for binomial response 
enetForecast_train = (enetResponse_train >=0.5);
%compute mean square error using train data
enetMSE_train = mean((enetForecast_train - trainTbl.loan_status).^2);
fprintf('Mean square error of the Elastic Net estimator using training sample is: %6.4f \n', enetMSE_train); 

% output confusion chart for training sample
figure()
chart_enet_train = confusionchart(trainTbl.loan_status, enetForecast_train);
title('Elastic Net confusion chart using training data')

%predict loan_status using test data
enetResponse_test = glmval(coef_enet, Predictor_testTbl{:,:}, 'logit'); % specify 'logit' for binomial response 
enetForecast_test = (enetResponse_test >=0.5);
%compute mean square error using test data
enetMSE_test = mean((enetForecast_test - testTbl.loan_status).^2);
fprintf('Mean square error of the Elastic Net estimator using testing sample is: %6.4f \n', enetMSE_test); 

% output confusion chart for testing sample
figure()
chart_enet_test = confusionchart(testTbl.loan_status, enetForecast_test);
title('Elastic Net confusion chart using testing data')

% -------------------------------------------------------------------------
% #########################################################################


%% bagging 
% fit an ensemble model using bagging algo (using a tree as the weak
% learner. output the estimated model as 'bagMdl'.)
t = templateTree('MaxNumSplits', 5, 'PredictorSelection','interaction-curvature','Reproducible', true); % using curvature algo when include categorical predictors
% #########################################################################
% -------------------------------------------------------------------------

numTree = 100; % number of trees in the ensemble
bagMdl = fitensemble(trainTbl(:,predictorNames), trainTbl.loan_status, 'Bag', numTree, t, 'Type', 'Classification');
% #########################################################################
                 
% forecast using training sample
bagForecast_train = predict(bagMdl,trainTbl(:,predictorNames));
bagMSE_train = mean((bagForecast_train - trainTbl.loan_status).^2);
fprintf('MSE of the bagging ensemble estimator using training sample is: %6.4f \n', bagMSE_train);  

% forecast using testing sample, and save MSE as bagMSE_test
% #########################################################################
% -------------------------------------------------------------------------
bagForecast_test = predict(bagMdl,testTbl(:,predictorNames));
bagMSE_test = mean((bagForecast_test - testTbl.loan_status).^2);

% -------------------------------------------------------------------------
% #########################################################################
fprintf('MSE of the bagging ensemble estimator using testing sample is: %6.4f \n', bagMSE_test); 

% predictor importance estimation 
% note 'PredictorSelection','interaction-curvature' ensures importance is
% not baised towards variables with many levels
imp = bagMdl.predictorImportance;
figure(); 
% #########################################################################
% -------------------------------------------------------------------------

bar(imp);
title('Predictor Importance Bagging');
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca
h.XTick = 1:length(bagMdl.PredictorNames);
h.XTickLabel = bagMdl.PredictorNames;
h.XTickLabelRotation = 45;  %this rotation makes the x variables easier to read
h.TickLabelInterpreter = 'none';
% -------------------------------------------------------------------------
% #########################################################################

% find top 10 important predictors
bagImportanceTbl = array2table(abs(imp)', 'RowNames', bagMdl.PredictorNames, 'VariableNames', {'abs_coef'});
bagImportanceTbl = sortrows(bagImportanceTbl, 'abs_coef', 'descend');
bagTop10 = bagImportanceTbl.Properties.RowNames(1:10);
disp(bagTop10)



%% boosting
% fit a boosting ensemble model using a tree as the weak learner.
% It is very similar to the procedure above. Save MSE for the training and
% testing sample as boostMSE_train and boostMSE_test, respectively. Find
% the top 10 important predictors.
% #########################################################################
% -------------------------------------------------------------------------

boostMdl = fitensemble(trainTbl, 'loan_status', 'AdaBoostM1', 100, 'Tree');

% Make predictions using the boosting ensemble model for the training and testing samples
boostForecast_train = predict(boostMdl, trainTbl);
boostForecast_test = predict(boostMdl, testTbl);

% Compute MSE for the training and testing samples
boostMSE_train = mean((boostForecast_train - trainTbl.loan_status).^2);
boostMSE_test = mean((boostForecast_test - testTbl.loan_status).^2);
fprintf('MSE of the boosting ensemble estimator using training sample is: %6.4f \n', boostMSE_train);
fprintf('MSE of the boosting ensemble estimator using testing sample is: %6.4f \n', boostMSE_test);
% Find the top 10 important predictors
importance = boostMdl.predictorImportance;
[~,idx] = sort(importance,'descend');
boosttop10 = predictorNames(idx(1:10));
disp('Top 10 important predictors using boosting ensemble estimator:');
disp(boosttop10);

% -------------------------------------------------------------------------
% #########################################################################



%% random forest   
% fit a random forest with 50 trees. Randomly select 1/3 of total
% predictors to build each tree. Name this model rfMdl.
% #########################################################################
% -------------------------------------------------------------------------

rng(1); % for reproducibility

ntrees = 50;
mtry = floor(numel(predictorNames)/3);

rfMdl = TreeBagger(ntrees, trainTbl, 'loan_status', 'Method', 'classification', ...
    'PredictorSelection', 'allsplits', 'OOBPrediction', 'on', 'OOBPredictorImportance', 'on', ...
    'MinLeafSize', 5, 'NumPrint', 10, 'MaxNumSplits', 100, 'NumPredictorsToSample', mtry);

% -------------------------------------------------------------------------
% #########################################################################
% MSE using training sample
rfForecast_train = predict(rfMdl, trainTbl(:,predictorNames)); % output is a cell array -> convert to numeric vector
rfForecast_train = cellfun(@str2double, rfForecast_train);  
rfMSE_train = mean((rfForecast_train - trainTbl.loan_status).^2);
fprintf('MSE of the random forest estimator using training sample is: %6.4f \n', rfMSE_train);

% MSE using testing sample
rfForecast_test = predict(rfMdl, testTbl(:,predictorNames)); 
rfForecast_test = cellfun(@str2double, rfForecast_test);  
rfMSE_test = mean((rfForecast_test - testTbl.loan_status).^2);
fprintf('MSE of the random forest estimator using testing sample is: %6.4f \n', rfMSE_test);

% Predictor importance estimation
% #########################################################################
% -------------------------------------------------------------------------

rng(1); % for reproducibility

ntrees = 50;
mtry = floor(numel(predictorNames)/3);

rfMdl = TreeBagger(ntrees, trainTbl, 'loan_status', 'Method', 'classification', ...
    'PredictorSelection', 'allsplits', 'OOBPrediction', 'on', 'OOBPredictorImportance', 'on', ...
    'MinLeafSize', 5, 'NumPrint', 10, 'MaxNumSplits', 100, 'NumPredictorsToSample', mtry);

% Estimate predictor importance
imp = rfMdl.OOBPermutedPredictorDeltaError;
fprintf('Estimate predictor importance is: %6.4f \n', imp);

% -------------------------------------------------------------------------
% #########################################################################

%% save data
save('assignment.mat', 'trainTbl','testTbl', 'top10', 'bottom10','glsMSE_train', 'glsMSE_test', ...
    'swMSE_train', 'swMSE_test', 'lassoMSE_train', 'lassoMSE_test', 'enetMSE_train', 'enetMSE_test',  ...
    'bagMSE_test', 'bagMSE_train', 'boostMSE_train', 'boostMSE_test', 'rfMSE_train', 'rfMSE_test'); 
    % change the .mat file name as firstname_surname_ID.mat.








