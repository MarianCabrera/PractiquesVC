clearvars,
close all,
clc,

ima=imread('https://www.ign.es/wms-inspire/pnoa-ma?REQUEST=GetMap&VERSION=1.1.0&SERVICE=WMS&SRS=EPSG:32631&BBOX=-2806.70,4664788.54,-2406.70,4665188.54&WIDTH=1024&HEIGHT=1024&LAYERS=OI.OrthoimageCoverage&STYLES=&FORMAT=JPEG&BGCOLOR=0xFFFFFF&TRANSPARENT=TRUE&EXCEPTION=INIMAGE');
imshow(ima)
im = rgb2gray(ima);

nPoints = 100;
% [arbre, noArbre] = getPointsImage(ima, nPoints);
% save('A-NA.mat','arbre', 'noArbre');

load('A-NA.mat');
 
indexA = randperm(nPoints);
indexNA = randperm(nPoints);

for i = 1: nPoints
    rA(i, :) = uint8(arbre(indexA(i), :));
    rNA(i, :) = uint8(noArbre(indexNA(i), :));
end
% save('randomsA-NA.mat','rA', 'rNA');
% load('randomsA-NA.mat')

% for i = 1 : nPoints
%     descA(i, :) = ima(rA(i, 1), rA(i, 2), :);
%     descNA(i, :) = ima(rNA(i, 1), rNA(i, 2), :);
% end

trainA = rA(1: 50, :);
trainNA = rNA(1: 50, :);
trainPoints = [trainA; trainNA];
trainLabels = zeros(nPoints, 1);
trainLabels(1 : 50) = 1;
trainLabels(51: 100) = 0;

testA = rA(51: 100, :);
testNA = rNA(51: 100, :);
test = [testA; testNA];
testLabels = zeros(nPoints, 1);
testLabels(1 : 50) = 1;
testLabels(51: 100) = 0;

for i = 1 : nPoints
    descTrain(i, 1:3) = double(ima(test(i, 2), test(i, 1), :));
    descTest(i, 1:3) = double(ima(test(i, 2), test(i, 1), :));
end
% 
% for i = 1 : size(points,2)
%     xCenter = points(1,i);
%     yCenter = points(2,i);
%     vals = im(yCenter - 2 : yCenter + 2, xCenter - 2 : xCenter + 2, :);
%     desc(i, :) = reshape(vals,1,[]);
% end

% SVM
% Mdl = fitcsvm(descTrain,trainLabels);
% predictLabels = resubPredict(Mdl,descTest);

% NAIVE BAYES
Mdl = fitcnb(descTrain,trainLabels);
predictLabels = predict(Mdl,descTest);

% K-NN
% Mdl = fitcknn(descTrain,trainLabels,'NumNeighbors',5,'Standardize',1);
% predictLabels = predict(Mdl,descTest);

cm = confusionchart(testLabels,predictLabels);
TP = cm.NormalizedValues(1,1);
FN = cm.NormalizedValues(1,2);
FP = cm.NormalizedValues(2,1);
TN = cm.NormalizedValues(2,2);

ACC =(TP + TN) / (50 + 50)

rgbVector = double(reshape(ima,size(ima,1)*size(ima,2),3));
predictLabels2 = predict(Mdl,rgbVector);

imNew = reshape(predictLabels2,size(im));
imshow(imNew)

function [arbre, noArbre] = getPointsImage(img, nPoints)
    figure(1),imshow(img)
    arbre = zeros(nPoints, 2);
    noArbre = zeros(nPoints, 2);
    for j=1:nPoints
        [x,y]=ginput(1);
        arbre(j, 1) = x;
        arbre(j, 2) = y;
%         arbre(j, 3) = 0;
        figure(1)
        hold on
        plot( arbre(j, 1), arbre(j, 2),'o','MarkerSize',5,'LineWidth', 1, 'Color', 'y');
        text(arbre(j, 1)+10,arbre(j, 2),string(j), 'Color', 'y');
        hold off
    end
    for j=1 :nPoints
        [x,y]=ginput(1);
        noArbre(j, 1) = x;
        noArbre(j, 2) = y;
%         noArbre(j, 3) = 1;
        figure(1)
        hold on
        plot( noArbre(j, 1), noArbre(j, 2),'o','MarkerSize',5,'LineWidth', 1, 'Color', 'm');
        text(noArbre(j, 1)+10,noArbre(j, 2),string(j), 'Color', 'm');
        hold off
    end
end



