clearvars,
close all,
clc,

im=imread("im1.jpg");
figure(1), imshow(im)
rgb=double(reshape(im,size(im,1)*size(im,2),3));

% % 2, 4, 8, 16
% K = 16; 
% 
% maxIter = 50; done = 0; loop = 0;
% clusters = initClusters(rgb, K);
% 
% while done == 0 && loop < maxIter
%     dist = getL2Dist(clusters, rgb, K);
%     labels = assignClusters(dist);
%     [clusters, done] = recalculateClusters(rgb, clusters, labels, K);
%     loop = loop + 1;
% end
% 
% imNew = reasignColors(im, rgb, clusters, labels, K);
% figure(2), imshow(uint8(imNew))


% HSV

hsv=double(reshape(rgb2hsv(im),size(im,1)*size(im,2),3));
hsv = hsv * 255;

% 2, 4, 8, 16
K = 16; 

maxIter = 50; done = 0; loop = 0;
clusters = initClusters(hsv, K);

while done == 0 && loop < maxIter
    dist = getRadDist(clusters, hsv, K);
    labels = assignClusters(dist);
    [clusters, done] = recalculateClusters(hsv, clusters, labels, K);
    loop = loop + 1;
end

imNew = reasignColors(im, hsv, clusters, labels, K) / 255;
imNew = hsv2rgb(imNew);
figure(2), imshow(imNew)

% ---------------------------------------------------------------

function c = initClusters(im, K)
    c = zeros(K, size(im, 2));
    for i = 1 : K, c(i, :) = im(randi(size(im, 1)), :); end
end

function dist = getRadDist(c, im, K)
    dist = zeros(K, size(im, 1));
    for i = 1 : K
        d1 = abs(c(i, 1) - im(:,1));
        d2 = abs(c(i, 2) - im(:,2));
        d3 = abs(c(i, 3) - im(:,3));
        
        if d1 > 127.5, d1 = 127.5 - (d1 - 127.5); end
        if d2 > 127.5, d2 = 127.5 - (d2 - 127.5); end
        if d3 > 127.5, d3 = 127.5 - (d3 - 127.5); end
        
        dist(i, :) = d1 + d2 + d3;
    end
end

function dist = getL2Dist(c, im, K)
    dist = zeros(K, size(im, 1));
    for i = 1 : K
        d1 = abs(c(i, 1) - im(:,1)).^2;
        d2 = abs(c(i, 2) - im(:,2)).^2;
        d3 = abs(c(i, 3) - im(:,3)).^2;
        dist(i, :) = sqrt(d1 + d2 + d3);
    end
end

function labels = assignClusters(dist)
    [M, labels] = min(dist, [], 1);
end

function [newC, done] = recalculateClusters(im, c, l, K)
    newC = c;
    done = 0;
    for i = 1 : K
        listPoints = find(l == i);
        sumR = 0; sumG = 0; sumB = 0;
        for j = listPoints
            sumR = sumR + im(j, 1);
            sumG = sumG + im(j, 2);
            sumB = sumB + im(j, 3);
        end
        num = size(listPoints,2);
        newC(i, 1) = uint8(sumR / num);
        newC(i, 2) = uint8(sumG / num);
        newC(i, 3) = uint8(sumB / num);
    end
    
    if isequal(newC, c), done = 1; end
end

function imNew = reasignColors(imOrig, im, c, l, K)
  imNew = im;
  for i = 1 : K
      listPoints = find(l == i);
      for j = listPoints
          imNew(j, :) = c(i, :);
      end
  end
  imNew = reshape(imNew,size(imOrig));
end




