%Question 2/MAP Error Minimization

%Setting seed (to make my work replicable)
rng(1881)

%First Section
%Clearing memory
clear all, close all,

%Number of feature dimensions
n = 2; 
%Number of iid samples
N = 400; 
%Class means
mu(:,1) = [0;0]; mu(:,2) = [3;3];
%Class variance-covariance matrices
Sigma(:,:,1) = [1 0;0 1]; Sigma(:,:,2) = [1 0;0 1];
%Class priors for labels 0 and 1 respectively
p = [0.5,0.5]; 
label = rand(1,N) >= p(1);
%Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))]; 
%Reserve space
x = zeros(n,N); 
%Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
figure(2), clf,
plot(x(1,label==0),x(2,label==0),'ob'), hold on,
plot(x(1,label==1),x(2,label==1),'+m'), axis equal,
legend('Class 0','Class 1'), 
title('Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q21distribution.png')

%Loss values (0-1 for this error minimization)
lambda = [0 1;1 0]; 
%Threshold
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); 
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma)
decision = (discriminantScore >= log(gamma));

%Probability of true negative
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); 
%Number of errors
error = p10*Nc(1)+p01*Nc(2);
%Probability of errors
proberror = [p10,p01]*Nc'/N;
disp('Probability of error')
disp(error)
disp(proberror)

%Class 0 gets a circle, class 1 gets a  +, correct green, incorrect red
figure(1), 
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,


%Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
%Plot equilevel contours of the discriminant function 
figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV,0,maxDSGV]); 
%Including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Model threshold' ), 
title('Data with Real Population/Class Labels and Decisions'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q21boundary.png')

%Second Section
%Clearing memory
clear all, close all,

%Number of feature dimensions
n = 2; 
%Number of iid samples
N = 400; 
%Class means
mu(:,1) = [0;0]; mu(:,2) = [3;3];
%Class variance-covariance matrices
Sigma(:,:,1) = [3 1;1 0.8]; Sigma(:,:,2) = [3 1;1 0.8];
%Class priors for labels 0 and 1 respectively
p = [0.5,0.5]; 
label = rand(1,N) >= p(1);
%Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))]; 
%Reserve space
x = zeros(n,N); 
%Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
figure(2), clf,
plot(x(1,label==0),x(2,label==0),'ob'), hold on,
plot(x(1,label==1),x(2,label==1),'+m'), axis equal,
legend('Class 0','Class 1'), 
title('Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q22distribution.png')

%Loss values (0-1 for this error minimization)
lambda = [0 1;1 0]; 
%Threshold
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); 
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma)
decision = (discriminantScore >= log(gamma));

%Probability of true negative
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); 
%Number of errors
error = p10*Nc(1)+p01*Nc(2);
%Probability of errors
proberror = [p10,p01]*Nc'/N;
disp('Probability of error')
disp(error)
disp(proberror)

%Class 0 gets a circle, class 1 gets a  +, correct green, incorrect red
figure(1), 
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

%Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
%Plot equilevel contours of the discriminant function 
figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV,0,maxDSGV]); 
%Including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Model threshold' ), 
title('Data with Real Population/Class Labels and Decisions'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q22boundary.png')

%Second Section
%Clearing memory
clear all, close all,

%Number of feature dimensions
n = 2; 
%Number of iid samples
N = 400; 
%Class means
mu(:,1) = [0;0]; mu(:,2) = [2;2];
%Class variance-covariance matrices
Sigma(:,:,1) = [2 0.5;0.5 1]; Sigma(:,:,2) = [2 -1.9;-1.9 5];
%Class priors for labels 0 and 1 respectively
p = [0.5,0.5]; 
label = rand(1,N) >= p(1);
%Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))]; 
%Reserve space
x = zeros(n,N); 
%Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
figure(2), clf,
plot(x(1,label==0),x(2,label==0),'ob'), hold on,
plot(x(1,label==1),x(2,label==1),'+m'), axis equal,
legend('Class 0','Class 1'), 
title('Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q23distribution.png')

%Loss values (0-1 for this error minimization)
lambda = [0 1;1 0]; 
%Threshold
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); 
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma)
decision = (discriminantScore >= log(gamma));

%Probability of true negative
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); 
%Number of errors
error = p10*Nc(1)+p01*Nc(2);
%Probability of errors
proberror = [p10,p01]*Nc'/N;
disp('Probability of error')
disp(error)
disp(proberror)

%Class 0 gets a circle, class 1 gets a  +, correct green, incorrect red
figure(1), 
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

%Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
%Plot equilevel contours of the discriminant function 
figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV,0,maxDSGV]); 
%Including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Model threshold' ), 
title('Data with Real Population/Class Labels and Decisions'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q23boundary.png')

%Fourth Section
%Clearing memory
clear all, close all,

%Number of feature dimensions
n = 2; 
%Number of iid samples
N = 400; 
%Class means
mu(:,1) = [0;0]; mu(:,2) = [3;3];
%Class variance-covariance matrices
Sigma(:,:,1) = [1 0;0 1]; Sigma(:,:,2) = [1 0;0 1];
%Class priors for labels 0 and 1 respectively
p = [0.05,0.95]; 
label = rand(1,N) >= p(1);
%Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))]; 
%Reserve space
x = zeros(n,N); 
%Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
figure(2), clf,
plot(x(1,label==0),x(2,label==0),'ob'), hold on,
plot(x(1,label==1),x(2,label==1),'+m'), axis equal,
legend('Class 0','Class 1'), 
title('Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q24distribution.png')

%Loss values (0-1 for this error minimization)
lambda = [0 1;1 0]; 
%Threshold
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); 
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma)
decision = (discriminantScore >= log(gamma));

%Probability of true negative
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); 
%Number of errors
error = p10*Nc(1)+p01*Nc(2);
%Probability of errors
proberror = [p10,p01]*Nc'/N;
disp('Probability of error')
disp(error)
disp(proberror)

%Class 0 gets a circle, class 1 gets a  +, correct green, incorrect red
figure(1), 
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

%Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
%Plot equilevel contours of the discriminant function 
figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV,0,maxDSGV]); 
%Including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Model threshold' ), 
title('Data with Real Population/Class Labels and Decisions'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q24boundary.png')

%Fifth Section
%Clearing memory
clear all, close all,

%Number of feature dimensions
n = 2; 
%Number of iid samples
N = 400; 
%Class means
mu(:,1) = [0;0]; mu(:,2) = [3;3];
%Class variance-covariance matrices
Sigma(:,:,1) = [3 1;1 0.8]; Sigma(:,:,2) = [3 1;1 0.8];
%Class priors for labels 0 and 1 respectively
p = [0.05,0.95]; 
label = rand(1,N) >= p(1);
%Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))]; 
%Reserve space
x = zeros(n,N); 
%Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
figure(2), clf,
plot(x(1,label==0),x(2,label==0),'ob'), hold on,
plot(x(1,label==1),x(2,label==1),'+m'), axis equal,
legend('Class 0','Class 1'), 
title('Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q25distribution.png')

%Loss values (0-1 for this error minimization)
lambda = [0 1;1 0]; 
%Threshold
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); 
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma)
decision = (discriminantScore >= log(gamma));

%Probability of true negative
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); 
%Number of errors
error = p10*Nc(1)+p01*Nc(2);
%Probability of errors
proberror = [p10,p01]*Nc'/N;
disp('Probability of error')
disp(error)
disp(proberror)

%Class 0 gets a circle, class 1 gets a  +, correct green, incorrect red
figure(1), 
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

%Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
%Plot equilevel contours of the discriminant function 
figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV,0,maxDSGV]); 
%Including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Model threshold' ), 
title('Data with Real Population/Class Labels and Decisions'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q25boundary.png')

%Sixth Section
%Clearing memory
clear all, close all,

%Number of feature dimensions
n = 2; 
%Number of iid samples
N = 400; 
%Class means
mu(:,1) = [0;0]; mu(:,2) = [2;2];
%Class variance-covariance matrices
Sigma(:,:,1) = [2 0.5;0.5 1]; Sigma(:,:,2) = [2 -1.9;-1.9 5];
%Class priors for labels 0 and 1 respectively
p = [0.05,0.95]; 
label = rand(1,N) >= p(1);
%Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))]; 
%Reserve space
x = zeros(n,N); 
%Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
figure(2), clf,
plot(x(1,label==0),x(2,label==0),'ob'), hold on,
plot(x(1,label==1),x(2,label==1),'+m'), axis equal,
legend('Class 0','Class 1'), 
title('Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q26distribution.png')

%Loss values (0-1 for this error minimization)
lambda = [0 1;1 0]; 
%Threshold
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); 
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma)
decision = (discriminantScore >= log(gamma));

%Probability of true negative
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); 
%Probability of false positive
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1);
%Probability of false negative
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2);
%Probability of true positive
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); 
%Number of errors
error = p10*Nc(1)+p01*Nc(2);
%Probability of errors
proberror = [p10,p01]*Nc'/N;
disp('Probability of error')
disp(error)
disp(proberror)

%Class 0 gets a circle, class 1 gets a  +, correct green, incorrect red
figure(1), 
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

%Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
%Plot equilevel contours of the discriminant function 
figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV,0,maxDSGV]); 
%Including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Model threshold' ), 
title('Data with Real Population/Class Labels and Decisions'),
xlabel('x_1'), ylabel('x_2'), 

%Save graph
saveas(gcf,'Q26boundary.png')

%Defining evalGaussian used in script
function g = evalGaussian(x,mu,Sigma)
%Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end