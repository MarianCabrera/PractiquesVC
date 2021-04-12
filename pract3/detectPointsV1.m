function main

    clearvars,
    close all,
    clc,

    path = "img/cnm/";
    left = imread(strcat(path, "image011.jpg"));
    center = imread(strcat(path, "image012.jpg"));
    right = imread(strcat(path, "image013.jpg"));

    left = rgb2gray(double(left)/255);
    center = rgb2gray(double(center)/255);
    right = rgb2gray(double(right)/255);

%     left = rgb2gray(left);
%     center = rgb2gray(center);
%     right = rgb2gray(right);

    points1 = detectSURFFeatures(left);
    points2 = detectSURFFeatures(center);
    [features1, points1] = extractFeatures(left, points1);
    [features2, points2] = extractFeatures(center, points2);

    indexPairs = matchFeatures(features1, features2, 'Method', 'Threshold');
    numMatchedPoints = int32(size(indexPairs, 1));

    matchedPoints1 = points1(indexPairs(:, 1), :);
    matchedPoints2 = points2(indexPairs(:, 2), :);

    [tform,inlierIdx] = estimateGeometricTransform2D(matchedPoints1,matchedPoints2,'projective');
    inlierPts1 = matchedPoints1(inlierIdx,:);
    inlierPts2  = matchedPoints2(inlierIdx,:);

    figure; showMatchedFeatures(left, center, inlierPts1.Location, inlierPts2.Location);

    outputView = imref2d(size(center));
    Ir = imwarp(left,tform,'OutputView',outputView);
    figure 
    imshow(Ir); 

    x1 = inlierPts1.Location(:,1);
    y1 = inlierPts1.Location(:,2);
    x2 = inlierPts2.Location(:,1);
    y2 = inlierPts2.Location(:,2);
    h = getDLT(x1, y1, x2, y2);
    c = getCornersToH(left, h);
    
    minX = min(c(:,1));
    minY = min(c(:,2));
    maxX = max(c(:,1));
    maxY = max(c(:,2));
    
    if minX > 0, minX = 0; end
    if minY > 0, minY = 0; end
    if maxX < size(center,2), maxX = size(center,2); end
    if maxY < size(center,1), maxY = size(center,1); end
    
    sizeY = maxY - minY;
    sizeX = maxX - minX;
    
    imOut = double(zeros(int32(sizeY), int32(sizeX)));
    imOutPlus = double(zeros(int32(sizeY), int32(sizeX)));
    imOut(int32((0 - minY) +1) : int32(size(center, 1) + (0 - minY)), int32((0 - minX)+1) : int32(size(center, 2) + (0 - minX))) = center(:,:);
    imOutPlus(int32((0 - minY) +1) : int32(size(center, 1) + (0 - minY)), int32((0 - minX)+1) : int32(size(center, 2) + (0 - minX))) = 1;
    
    invHL = inv(h);
    
    
    for j = 1 : size(imOut, 2)
        for i = 1 : size(imOut, 1)
            xToH = j + minX;
            yToH = i + minY;
            pL = applyDLT(xToH, yToH, invHL);
            if int32(pL(1))>0 && int32(pL(1)) < size(center, 2) && int32(pL(2))>0 && int32(pL(2)) < size(center, 1)
                imOut(i, j) = double(imOut(i, j)) + double(left(int32(pL(2)), int32(pL(1))));
                imOutPlus(i, j) =  imOutPlus(i, j) + 1;
            end
%             pR = applyDLT(xToH, yToH, invHR);
%             if int32(pR(1))>0 && int32(pR(1)) < size(im, 2) && int32(pR(2))>0 && int32(pR(2)) < size(im, 1)
%                 imOut(i, j, :) = double(imOut(i, j, :)) + double(imR(int32(pR(2)), int32(pR(1)), :));
%                 imOutPlus(i, j) =  imOutPlus(i, j) + 1;
%             end
        end
    end
    
    for j = 1 : size(imOut, 2)
        for i = 1 : size(imOut, 1)
            if(imOutPlus(i, j) > 1)
                n = imOutPlus(i, j);
                imOut(i, j) = double(imOut(i, j)) ./ n;
            end
        end
    end
    
    
    
    
    figure(3)
    imshow(imOut)
    
    
    

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