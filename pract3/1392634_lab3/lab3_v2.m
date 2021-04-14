function main
    close all,

%     path = "img/ee/";
%     left = imread(strcat(path, "image006.jpg"));
%     center = imread(strcat(path, "image007.jpg"));
%     right = imread(strcat(path, "image008.jpg"));

    path = "img/meves/";
    left = imread(strcat(path, "3.jpg"));
    center = imread(strcat(path, "2.jpg"));
    right = imread(strcat(path, "1.jpg"));

    left = double(left)/255;
    center = double(center)/255;
    right = double(right)/255;

    nPoints = 4;
    imOut = getImage(left, center, right, nPoints);
    figure(3)
    imshow(imOut)
end

function [x1, y1] = getPointsImage(img, nPoints, n)
    figure(n),imshow(img)
    enableDefaultInteractivity(gca);
    x1=[]; y1=[];
    for j=1:nPoints
        [x1(j),y1(j)]=ginput(1);
        figure(n)
        hold on
        plot(x1(j),y1(j),'rx','MarkerSize',20,'LineWidth',3);
        text(x1(j)-10,y1(j)-50,string(j));
        hold off
    end
end

function [x1, y1, x2, y2] = getSURF(im1, im2)
    im1 = rgb2gray(im1);
    im2 = rgb2gray(im2);
    
     points1 = detectSURFFeatures(im1);
     points2 = detectSURFFeatures(im2);
%      numPoints = 100;
%      points1 = selectUniform(points1,numPoints,size(im1));
%      points2 = selectUniform(points2,numPoints,size(im2));
     
    [features1, points1] = extractFeatures(im1, points1);
    [features2, points2] = extractFeatures(im2, points2);
    
    indexPairs = matchFeatures(features1, features2, 'Unique', true);
    matchedPoints1 = points1(indexPairs(:, 1), :);
    matchedPoints2 = points2(indexPairs(:, 2), :);

    [tform,inlierIm] = estimateGeometricTransform2D(matchedPoints1,matchedPoints2,'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    inlierPts1 = matchedPoints1(inlierIm,:);
    inlierPts2  = matchedPoints2(inlierIm,:);

    figure; showMatchedFeatures(im1, im2, inlierPts1.Location, inlierPts2.Location, 'montage');
    
    x1 = inlierPts1.Location(:,1);
    y1 = inlierPts1.Location(:,2);
    x2 = inlierPts2.Location(:,1);
    y2 = inlierPts2.Location(:,2);
end

function H = getDLT(x1, y1, x2, y2, points)
    M = [];
    for i=1:points
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

function imOut = getImage(imL, im, imR, nPoints)
    type = 1;
    switch type
        case 0
            [xL, yL] = getPointsImage(imL, nPoints, 1);
            [xCL, yCL] = getPointsImage(im, nPoints, 2);
            [xR, yR] = getPointsImage(imR, nPoints, 3);
            [xCR, yCR] = getPointsImage(im, nPoints, 4);
            close all,
            hL = getDLT(xL, yL, xCL, yCL, nPoints);
            hR = getDLT(xR, yR, xCR, yCR, nPoints);
        case 1
            [xL, yL, xCL, yCL] = getSURF(imL, im);
            [xCR, yCR, xR, yR] = getSURF(im, imR);
            sizeL = size(xL, 1);
            sizeR = size(xR, 1);
            hL = getDLT(xL, yL, xCL, yCL, sizeL);
            hR = getDLT(xR, yR, xCR, yCR, sizeR);
    end
  
    cL = getCornersToH(imL, hL);
    cR = getCornersToH(imR, hR);
    c = [cL; cR];
    
    minX = min(c(:,1));
    minY = min(c(:,2));
    maxX = max(c(:,1));
    maxY = max(c(:,2));
    
    if minX > 0, minX = 0; end
    if minY > 0, minY = 0; end
    if maxX < size(im,2), maxX = size(im,2); end
    if maxY < size(im,1), maxY = size(im,1); end
    
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
            xToH = j + minX+1;
            yToH = i + minY+1;
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
