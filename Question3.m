%Question 3/Fisher LDA Classifier

%Setting seed (to make my work replicable)
rng(1881)

%First section
%Clearing memory
clear all, close all,

%Number of samples
N = 400; 
%The means for the first distribution
mu(:,1) = [0;0]; 
%The means for the second distribution
mu(:,2) = [3;3];
%The variance-covariance matrix of the first distribution
Sigma(:,:,1) = [1 0;0 1]; 
%The variance-covariance matrix of the second distribution 
Sigma(:,:,2) = [1 0;0 1];
%Class priors for the first class and the second class
p = [0.5,0.5]; 
%Generating number of random variables to run through each pdf (samples from each label)
label = rand(1,N)>=p(1);
%Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))];
%Create space for samples
x = zeros(2,N); 
%Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

%Fisher LDA Error Minimization
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
%LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
[V,D] = eig(inv(Sw)*Sb); 
[~,ind] = sort(diag(D),'descend');
%Fisher LDA projection vector
wLDA = V(:,ind(1)); 
%All data projected on to the line spanned by wLDA
yLDA = wLDA'*x; 
%Ensuring class 1 falls on the + side of the axis
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; 
%Flipping yLDA accordingly
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; 
figure(1), clf,
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'ob'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+m'), axis equal,
legend('Class 0','Class 1'), 
title('LDA Projection of Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q31distribution.png')

%Finding the threshold (I will just get the best number, not the median of the two best)
for i = 1:N
%Assuming threshold is yLDA
decision(:,i) = (yLDA >= yLDA(i));
dec = transpose(decision(:,i));
%Probability of true negative
ind00 = find(dec==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(dec==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(dec==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(dec==1 & label==1); p11 = length(ind11)/Nc(2); 
%Number of errors
error(i) = length(ind10)+length(ind01);
%Probability of errors
proberror(i) = error(i)/N;
clear ind00 ind01 ind10 ind11 p00 p01 p10 p11 dec
end

%Finding the row number of the minimum error producing threshold choice
[t1,t2] = min(error)
disp('Probability of error')
disp(error(t2))
disp(proberror(t2))

%Decisions of the error minimizing threshold
dec = transpose(decision(:,t2));
%Probability of true negative
ind00 = find(dec==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(dec==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(dec==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(dec==1 & label==1); p11 = length(ind11)/Nc(2); 


%Class 0 gets a circle, class 1 gets a  +, correct green, incorrect red
figure(2), 
plot(yLDA(ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(ind01),zeros(1,length(ind01)),'+r'); hold on,
plot(yLDA(ind11),zeros(1,length(ind11)),'+g'); hold on,
%Threshold is the yLDA at the error minimizing point
xline(yLDA(t2))
axis equal,
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Model threshold' ), 
title('Data with Real Population/Class Labels and Decisions'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q31boundary.png')

%Second section
%Clearing memory
clear all, close all,

%Number of samples
N = 400; 
%The means for the first distribution
mu(:,1) = [0;0]; 
%The means for the second distribution
mu(:,2) = [3;3];
%The variance-covariance matrix of the first distribution
Sigma(:,:,1) = [3 1;1 0.8]; 
%The variance-covariance matrix of the second distribution 
Sigma(:,:,2) = [3 1;1 0.8];
%Class priors for the first class and the second class
p = [0.5,0.5]; 
%Generating number of random variables to run through each pdf (samples from each label)
label = rand(1,N)>=p(1);
%Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))];
%Create space for samples
x = zeros(2,N); 
%Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

%Fisher LDA Error Minimization
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
%LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
[V,D] = eig(inv(Sw)*Sb); 
[~,ind] = sort(diag(D),'descend');
%Fisher LDA projection vector
wLDA = V(:,ind(1)); 
%All data projected on to the line spanned by wLDA
yLDA = wLDA'*x; 
%Ensuring class 1 falls on the + side of the axis
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; 
%Flipping yLDA accordingly
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; 
figure(1), clf,
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'ob'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+m'), axis equal,
legend('Class 0','Class 1'), 
title('LDA Projection of Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q32distribution.png')

%Finding the threshold (I will just get the best number, not the median of the two best)
for i = 1:N
%Assuming threshold is yLDA
decision(:,i) = (yLDA >= yLDA(i));
dec = transpose(decision(:,i));
%Probability of true negative
ind00 = find(dec==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(dec==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(dec==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(dec==1 & label==1); p11 = length(ind11)/Nc(2); 
%Number of errors
error(i) = length(ind10)+length(ind01);
%Probability of errors
proberror(i) = error(i)/N;
clear ind00 ind01 ind10 ind11 p00 p01 p10 p11 dec
end

%Finding the row number of the minimum error producing threshold choice
[t1,t2] = min(error)
disp('Probability of error')
disp(error(t2))
disp(proberror(t2))

%Decisions of the error minimizing threshold
dec = transpose(decision(:,t2));
%Probability of true negative
ind00 = find(dec==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(dec==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(dec==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(dec==1 & label==1); p11 = length(ind11)/Nc(2); 


%Class 0 gets a circle, class 1 gets a  +, correct green, incorrect red
figure(2), 
plot(yLDA(ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(ind01),zeros(1,length(ind01)),'+r'); hold on,
plot(yLDA(ind11),zeros(1,length(ind11)),'+g'); hold on,
%Threshold is the yLDA at the error minimizing point
xline(yLDA(t2))
axis equal,
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Model threshold' ), 
title('Data with Real Population/Class Labels and Decisions'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q32boundary.png')

%Third section
%Clearing memory
clear all, close all,

%Number of samples
N = 400; 
%The means for the first distribution
mu(:,1) = [0;0]; 
%The means for the second distribution
mu(:,2) = [2;2];
%The variance-covariance matrix of the first distribution
Sigma(:,:,1) = [2 0.5;0.5 1]; 
%The variance-covariance matrix of the second distribution 
Sigma(:,:,2) = [2 -1.9;-1.9 5];
%Class priors for the first class and the second class
p = [0.5,0.5]; 
%Generating number of random variables to run through each pdf (samples from each label)
label = rand(1,N)>=p(1);
%Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))];
%Create space for samples
x = zeros(2,N); 
%Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

%Fisher LDA Error Minimization
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
%LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
[V,D] = eig(inv(Sw)*Sb); 
[~,ind] = sort(diag(D),'descend');
%Fisher LDA projection vector
wLDA = V(:,ind(1)); 
%All data projected on to the line spanned by wLDA
yLDA = wLDA'*x; 
%Ensuring class 1 falls on the + side of the axis
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; 
%Flipping yLDA accordingly
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; 
figure(1), clf,
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'ob'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+m'), axis equal,
legend('Class 0','Class 1'), 
title('LDA Projection of Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q33distribution.png')

%Finding the threshold (I will just get the best number, not the median of the two best)
for i = 1:N
%Assuming threshold is yLDA
decision(:,i) = (yLDA >= yLDA(i));
dec = transpose(decision(:,i));
%Probability of true negative
ind00 = find(dec==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(dec==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(dec==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(dec==1 & label==1); p11 = length(ind11)/Nc(2); 
%Number of errors
error(i) = length(ind10)+length(ind01);
%Probability of errors
proberror(i) = error(i)/N;
clear ind00 ind01 ind10 ind11 p00 p01 p10 p11 dec
end

%Finding the row number of the minimum error producing threshold choice
[t1,t2] = min(error)
disp('Probability of error')
disp(error(t2))
disp(proberror(t2))

%Decisions of the error minimizing threshold
dec = transpose(decision(:,t2));
%Probability of true negative
ind00 = find(dec==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(dec==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(dec==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(dec==1 & label==1); p11 = length(ind11)/Nc(2); 


%Class 0 gets a circle, class 1 gets a  +, correct green, incorrect red
figure(2), 
plot(yLDA(ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(ind01),zeros(1,length(ind01)),'+r'); hold on,
plot(yLDA(ind11),zeros(1,length(ind11)),'+g'); hold on,
%Threshold is the yLDA at the error minimizing point
xline(yLDA(t2))
axis equal,
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Model threshold' ), 
title('Data with Real Population/Class Labels and Decisions'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q33boundary.png')

%Fourth section
%Clearing memory
clear all, close all,

%Number of samples
N = 400; 
%The means for the first distribution
mu(:,1) = [0;0]; 
%The means for the second distribution
mu(:,2) = [3;3];
%The variance-covariance matrix of the first distribution
Sigma(:,:,1) = [1 0;0 1]; 
%The variance-covariance matrix of the second distribution 
Sigma(:,:,2) = [1 0;0 1];
%Class priors for the first class and the second class
p = [0.05,0.95]; 
%Generating number of random variables to run through each pdf (samples from each label)
label = rand(1,N)>=p(1);
%Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))];
%Create space for samples
x = zeros(2,N); 
%Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

%Fisher LDA Error Minimization
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
%LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
[V,D] = eig(inv(Sw)*Sb); 
[~,ind] = sort(diag(D),'descend');
%Fisher LDA projection vector
wLDA = V(:,ind(1)); 
%All data projected on to the line spanned by wLDA
yLDA = wLDA'*x; 
%Ensuring class 1 falls on the + side of the axis
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; 
%Flipping yLDA accordingly
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; 
figure(1), clf,
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'ob'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+m'), axis equal,
legend('Class 0','Class 1'), 
title('LDA Projection of Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q34distribution.png')

%Finding the threshold (I will just get the best number, not the median of the two best)
for i = 1:N
%Assuming threshold is yLDA
decision(:,i) = (yLDA >= yLDA(i));
dec = transpose(decision(:,i));
%Probability of true negative
ind00 = find(dec==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(dec==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(dec==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(dec==1 & label==1); p11 = length(ind11)/Nc(2); 
%Number of errors
error(i) = length(ind10)+length(ind01);
%Probability of errors
proberror(i) = error(i)/N;
clear ind00 ind01 ind10 ind11 p00 p01 p10 p11 dec
end

%Finding the row number of the minimum error producing threshold choice
[t1,t2] = min(error)
disp('Probability of error')
disp(error(t2))
disp(proberror(t2))

%Decisions of the error minimizing threshold
dec = transpose(decision(:,t2));
%Probability of true negative
ind00 = find(dec==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(dec==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(dec==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(dec==1 & label==1); p11 = length(ind11)/Nc(2); 


%Class 0 gets a circle, class 1 gets a  +, correct green, incorrect red
figure(2), 
plot(yLDA(ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(ind01),zeros(1,length(ind01)),'+r'); hold on,
plot(yLDA(ind11),zeros(1,length(ind11)),'+g'); hold on,
%Threshold is the yLDA at the error minimizing point
xline(yLDA(t2))
axis equal,
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Model threshold' ), 
title('Data with Real Population/Class Labels and Decisions'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q34boundary.png')

%Fifth section
%Clearing memory
clear all, close all,

%Number of samples
N = 400; 
%The means for the first distribution
mu(:,1) = [0;0]; 
%The means for the second distribution
mu(:,2) = [3;3];
%The variance-covariance matrix of the first distribution
Sigma(:,:,1) = [3 1;1 0.8]; 
%The variance-covariance matrix of the second distribution 
Sigma(:,:,2) = [3 1;1 0.8];
%Class priors for the first class and the second class
p = [0.05,0.95]; 
%Generating number of random variables to run through each pdf (samples from each label)
label = rand(1,N)>=p(1);
%Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))];
%Create space for samples
x = zeros(2,N); 
%Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

%Fisher LDA Error Minimization
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
%LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
[V,D] = eig(inv(Sw)*Sb); 
[~,ind] = sort(diag(D),'descend');
%Fisher LDA projection vector
wLDA = V(:,ind(1)); 
%All data projected on to the line spanned by wLDA
yLDA = wLDA'*x; 
%Ensuring class 1 falls on the + side of the axis
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; 
%Flipping yLDA accordingly
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; 
figure(1), clf,
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'ob'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+m'), axis equal,
legend('Class 0','Class 1'), 
title('LDA Projection of Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q35distribution.png')

%Finding the threshold (I will just get the best number, not the median of the two best)
for i = 1:N
%Assuming threshold is yLDA
decision(:,i) = (yLDA >= yLDA(i));
dec = transpose(decision(:,i));
%Probability of true negative
ind00 = find(dec==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(dec==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(dec==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(dec==1 & label==1); p11 = length(ind11)/Nc(2); 
%Number of errors
error(i) = length(ind10)+length(ind01);
%Probability of errors
proberror(i) = error(i)/N;
clear ind00 ind01 ind10 ind11 p00 p01 p10 p11 dec
end

%Finding the row number of the minimum error producing threshold choice
[t1,t2] = min(error)
disp('Probability of error')
disp(error(t2))
disp(proberror(t2))

%Decisions of the error minimizing threshold
dec = transpose(decision(:,t2));
%Probability of true negative
ind00 = find(dec==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(dec==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(dec==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(dec==1 & label==1); p11 = length(ind11)/Nc(2); 


%Class 0 gets a circle, class 1 gets a  +, correct green, incorrect red
figure(2), 
plot(yLDA(ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(ind01),zeros(1,length(ind01)),'+r'); hold on,
plot(yLDA(ind11),zeros(1,length(ind11)),'+g'); hold on,
%Threshold is the yLDA at the error minimizing point
xline(yLDA(t2))
axis equal,
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Model threshold' ), 
title('Data with Real Population/Class Labels and Decisions'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q35boundary.png')

%Sixth section
%Clearing memory
clear all, close all,

%Number of samples
N = 400; 
%The means for the first distribution
mu(:,1) = [0;0]; 
%The means for the second distribution
mu(:,2) = [2;2];
%The variance-covariance matrix of the first distribution
Sigma(:,:,1) = [2 0.5;0.5 1]; 
%The variance-covariance matrix of the second distribution 
Sigma(:,:,2) = [2 -1.9;-1.9 5];
%Class priors for the first class and the second class
p = [0.05,0.95]; 
%Generating number of random variables to run through each pdf (samples from each label)
label = rand(1,N)>=p(1);
%Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))];
%Create space for samples
x = zeros(2,N); 
%Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

%Fisher LDA Error Minimization
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
%LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
[V,D] = eig(inv(Sw)*Sb); 
[~,ind] = sort(diag(D),'descend');
%Fisher LDA projection vector
wLDA = V(:,ind(1)); 
%All data projected on to the line spanned by wLDA
yLDA = wLDA'*x; 
%Ensuring class 1 falls on the + side of the axis
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; 
%Flipping yLDA accordingly
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; 
figure(1), clf,
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'ob'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+m'), axis equal,
legend('Class 0','Class 1'), 
title('LDA Projection of Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q36distribution.png')

%Finding the threshold (I will just get the best number, not the median of the two best)
for i = 1:N
%Assuming threshold is yLDA
decision(:,i) = (yLDA >= yLDA(i));
dec = transpose(decision(:,i));
%Probability of true negative
ind00 = find(dec==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(dec==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(dec==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(dec==1 & label==1); p11 = length(ind11)/Nc(2); 
%Number of errors
error(i) = length(ind10)+length(ind01);
%Probability of errors
proberror(i) = error(i)/N;
clear ind00 ind01 ind10 ind11 p00 p01 p10 p11 dec
end

%Finding the row number of the minimum error producing threshold choice
[t1,t2] = min(error)
disp('Probability of error')
disp(error(t2))
disp(proberror(t2))

%Decisions of the error minimizing threshold
dec = transpose(decision(:,t2));
%Probability of true negative
ind00 = find(dec==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(dec==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(dec==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(dec==1 & label==1); p11 = length(ind11)/Nc(2); 


%Class 0 gets a circle, class 1 gets a  +, correct green, incorrect red
figure(2), 
plot(yLDA(ind00),zeros(1,length(ind00)),'og'); hold on,
plot(yLDA(ind10),zeros(1,length(ind10)),'or'); hold on,
plot(yLDA(ind01),zeros(1,length(ind01)),'+r'); hold on,
plot(yLDA(ind11),zeros(1,length(ind11)),'+g'); hold on,
%Threshold is the yLDA at the error minimizing point
xline(yLDA(t2))
axis equal,
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Model threshold' ), 
title('Data with Real Population/Class Labels and Decisions'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q36boundary.png')