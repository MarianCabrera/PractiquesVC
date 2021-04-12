clearvars,
close all,
clc,

path = "img/cnm/";
left = imread(strcat(path, "image011.jpg"));
center = imread(strcat(path, "image012.jpg"));
right = imread(strcat(path, "image013.jpg"));

 left = double(left)/255;
 center = double(center)/255;
 right = double(right)/255;

imOut =  getImage(left, center, right);
figure(3)
imshow(imOut)

function [p1, p2] = harris(im1, im2)
    gC = rgb2gray(im1);
    gL = rgb2gray(im2);
    points1 = detectHarrisFeatures(gC);
    points2 = detectHarrisFeatures(gL);
    
    [features1,valid_points1] = extractFeatures(gC,points1);
    [features2,valid_points2] = extractFeatures(gL,points2);
    
    indexPairs = matchFeatures(features1,features2);
    matchedPoints1 = valid_points1(indexPairs(:,1),:);
    matchedPoints2 = valid_points2(indexPairs(:,2),:);
    
    p1 = matchedPoints1.Location;
    p2 = matchedPoints2.Location;
    
    figure; 
    showMatchedFeatures(gC,gL,matchedPoints1,matchedPoints2,'montage');
end

function [p1, p2] = surf(im1, im2)
    gL = rgb2gray(im1);
    gC = rgb2gray(im2);
    points1 = detectSURFFeatures(gC);
    points2 = detectSURFFeatures(gL);

    [f1,vpts1] = extractFeatures(gC,points1);
    [f2,vpts2] = extractFeatures(gL,points2);

    indexPairs = matchFeatures(f1,f2) ;
    matchedPoints1 = vpts1(indexPairs(:,1));
    matchedPoints2 = vpts2(indexPairs(:,2));
    
%     [tform,inlierIdx] = estimateGeometricTransform2D(matchedPoints1,matchedPoints2,'similarity');
%     inlierPts1 = matchedPoints1(inlierIdx,:);
%     inlierPts2 = matchedPoints2(inlierIdx,:);
    
%     p1 = inlierPts1.Location;
%     p2 = inlierPts2.Location;
    p1 = matchedPoints1.Location;
    p2 = matchedPoints2.Location;

    figure; 
    showMatchedFeatures(gC,gL,p1,p2,'montage');
    
%     h = estimateFundamentalMatrix(matchedPoints1, matchedPoints2,'Method','RANSAC','NumTrials',2000,'DistanceThreshold',1e-4);
end

function H = getDLT(x1, y1, x2, y2)
    M = [];
    for i=1:4
        M = [ M ;
        x1(i) y1(i) 1 0 0 0 -x2(i)*x1(i) -x2(i)*y1(i) -x2(i);
        0 0 0 x1(i) y1(i) 1 -y2(i)*x1(i) -y2(i)*y1(i) -y2(i)];
    end
    [u,s,v] = svd( M );
    H = reshape( v(:,end), 3, 3 )';
    H = H / H(3,3);
end

function p = applyDLT(x, y, H)
    p = H*[x y 1]';
    p = p/p(3);
end

function c = getCornersToH(im, h)
    s = size(im);
    corners = [0 0; s(2) 0; s(2) s(1); 0 s(1)];
    c = zeros(4,2);
    for i = 1: 4
        p = applyDLT(corners(i, 1), corners(i, 2), h);
        c(i, 1) = p(1);
        c(i, 2) = p(2);
    end
end

function imOut = getImage(imL, im, imR)
%     [xL, yL] = getPointsImage(imL, nPoints, 1);
%     [xCL, yCL] = getPointsImage(im, nPoints, 2);
%     hL = getDLT(xL, yL, xCL, yCL);

    [p1, p2] = harris(im, imL);
%     hL = getDLT(p1(:, 1), p1(:, 2), p2(:, 1), p2(:, 2));
    hL = estimateFundamentalMatrix(p1, p2,'Method','RANSAC', 'NumTrials',2000,'DistanceThreshold',1e-4);
%     [xR, yR] = getPointsImage(imR, nPoints, 3);
%     [xCR, yCR] = getPointsImage(im, nPoints, 4);
%     hR = getDLT(xR, yR, xCR, yCR);
%     close all,
     [p1, p2] = harris(im, imR);
%     hR = getDLT(p1(:, 1), p1(:, 2), p2(:, 1), p2(:, 2));
    hR = estimateFundamentalMatrix(p1, p2,'Method','RANSAC', 'NumTrials',2000,'DistanceThreshold',1e-4);

    cL = getCornersToH(imL, hL);
    cR = getCornersToH(imR, hR);

    c = [cL; cR];
    
    minX = min(c(:,1));
    minY = min(c(:,2));
    maxX = max(c(:,1));
    maxY = max(c(:,2));
    
    sizeY = maxY - minY;
    sizeX = maxX - minX;
    
    imOut = double(zeros(int32(sizeY), int32(sizeX), 3));
    imOutPlus = double(zeros(int32(sizeY), int32(sizeX)));
    imOut(int32((0 - minY) +1) : int32(size(im, 1) + (0 - minY)), int32((0 - minX)+1) : int32(size(im, 2) + (0 - minX)), :) = im(:,:,:);
    imOutPlus(int32((0 - minY) +1) : int32(size(im, 1) + (0 - minY)), int32((0 - minX)+1) : int32(size(im, 2) + (0 - minX))) = 1;
   
    
    invHL = inv(hL);
    invHR = inv(hR);
    
    for j = 1 : size(imOut, 2)
        for i = 1 : size(imOut, 1)
            xToH = j + minX;
            yToH = i + minY;
            pL = applyDLT(xToH, yToH, invHL);
            if int32(pL(1))>0 && int32(pL(1)) < size(im, 2) && int32(pL(2))>0 && int32(pL(2)) < size(im, 1)
                imOut(i, j, :) = double(imOut(i, j, :)) + double(imL(int32(pL(2)), int32(pL(1)), :));
                imOutPlus(i, j) =  imOutPlus(i, j) + 1;
            end
            pR = applyDLT(xToH, yToH, invHR);
            if int32(pR(1))>0 && int32(pR(1)) < size(im, 2) && int32(pR(2))>0 && int32(pR(2)) < size(im, 1)
                imOut(i, j, :) = double(imOut(i, j, :)) + double(imR(int32(pR(2)), int32(pR(1)), :));
                imOutPlus(i, j) =  imOutPlus(i, j) + 1;
            end
        end
    end
    
    for j = 1 : size(imOut, 2)
        for i = 1 : size(imOut, 1)
            if(imOutPlus(i, j) > 1)
                n = imOutPlus(i, j);
                imOut(i, j, :) = double(imOut(i, j, :)) ./ n;
            end
        end
    end
end