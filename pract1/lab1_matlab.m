% INIT 
clearvars,
close all,
clc,
addpath('lab1\highway\input')
addpath('lab1\highway\groundtruth')

% -- TASCA 1 -- 
allFiles = dir('lab1\highway\input\*.jpg');
files = allFiles(1051:1350);

tFiles = files(1:150);
training = {tFiles.name};
sFiles = files(151:end);
segmenting = {sFiles.name};
for i = 1:150
    trainingIMG{i}  = rgb2gray(imread(training{i}));
    segmentingIMG{i}  = rgb2gray(imread(segmenting{i}));
end
trainingGroup = double(cat(3, trainingIMG{:}))./256;
segmentingGroup = double(cat(3, segmentingIMG{:}))./256;
 
% -- TASCA 2 --
meanIMG = mean(trainingGroup, 3, 'native');
stdIM = std(trainingGroup,0,3);

% -- TASCA 3 --
result = zeros(240,320,150);
thr = 0.2;
for i = 1 : 150
  img = abs(meanIMG - segmentingGroup(:,:,i));
  result(:,:,i) = (img > thr); 
end

% -- TASCA 4 --
resultV2 = zeros(240,320,150);
%a = 0.7;
%b = 0.2;
a = 0.13;
b = 0.16;
thr = a.*stdIM + b;
for i = 1 : 150
  img = abs(meanIMG - segmentingGroup(:,:,i));
  resultV2(:,:,i) = (img > thr); 
end

% -- TASCA 5 --
video = VideoWriter('lab1VideoMatlab.avi');
open(video);
for i = 1 : 150
    writeVideo(video, resultV2(:,:,i));
end
close(video);

% -- TASCA 6 --
allGroundtruth = dir('lab1\highway\groundtruth\*.png');
groundtruth = allGroundtruth(1051:1350);
gFiles = groundtruth(151:end);
ground = {gFiles.name};
for i = 1:150
    groundIMG{i}  = imread(ground{i});
end
groundGroup = double(cat(3, groundIMG{:}))./256;
groundGroup(:,:,:) = groundGroup(:,:,:) > 0;

errorScore = 0;
for i = 1:150
    compGroup = cat(3,groundGroup(:,:,i), resultV2(:,:,i));
    score = std(compGroup,0,3);
    errorScore = errorScore + sum(score, 'all');
end
errorScore = errorScore / 150

% bestErrorScore = errorScore;
% bestA = 0;
% bestB = 0;
% for a = 0 : 0.01 : 2
%     for b = 0: 0.01 : 2
%         errorScore = 0;
%         for i = 1 : 150
%             img = abs(meanIMG - segmentingGroup(:,:,i));
%             resultV2(:,:,i) = (img > (a.*stdIM + b));
%             compGroup = cat(3,groundGroup(:,:,i), resultV2(:,:,i));
%             score = std(compGroup,0,3);
%             errorScore = errorScore + sum(score, 'all');
%         end
%         errorScore = errorScore / 150;
%         if errorScore < bestErrorScore
%             bestA = a;
%             bestB = b;
%             bestErrorScore = errorScore;
%         end
%     end
% end