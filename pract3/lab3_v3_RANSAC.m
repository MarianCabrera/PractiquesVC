path = "img/cnm/";
left = imread(strcat(path, "image011.jpg"));
center = imread(strcat(path, "image012.jpg"));
right = imread(strcat(path, "image013.jpg"));
grayC = rgb2gray(center);
grayL = rgb2gray(left);

left = double(left)/255;
center = double(center)/255;
right = double(right)/255;


% [p1, p2]  =getRANSAC(grayC, grayL);

imOut =  getImage(left, center, right);
imshow(imOut)

% function [x, y] = getPointsImage(img, nPoints, n)
%     figure(n),imshow(img)
%     [x, y] = ginput(nPoints);
% end

function [p1, p2]= getRANSAC(im, imAdd)
ptsIm  = detectSURFFeatures(im);
ptsImAdd = detectSURFFeatures(imAdd);

[featuresIm,validPtsIm] = extractFeatures(im,ptsIm);
[featuresImAdd,validPtsImAdd] = extractFeatures(imAdd,ptsImAdd);

index_pairs = matchFeatures(featuresIm,featuresImAdd);
matchedPtsIm  = validPtsIm(index_pairs(:,1));
matchedPtsImAdd = validPtsImAdd(index_pairs(:,2));

[tform,inlierIdx] = estimateGeometricTransform2D(matchedPtsImAdd,matchedPtsIm,'similarity');
inlierPtsImAdd = matchedPtsImAdd(inlierIdx,:);
inlierPtsIm  = matchedPtsIm(inlierIdx,:);

p1 = inlierPtsIm.Location;
p2 = inlierPtsImAdd.Location;

end

function H = getDLT(x1, y1, x2, y2, nPoints)
    M = [];
    for i=1:nPoints
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
    [p1, p2]= getRANSAC(rgb2gray(im), rgb2gray(imL));
    nPointsL = size(p1, 1);
    hL = getDLT(p1(:,1), p2(:,1), p1(:,2), p2(:,2),nPointsL);
    
%     [xR, yR] = getPointsImage(imR, nPoints, 3);
%     [xCR, yCR] = getPointsImage(im, nPoints, 4);
%     hR = getDLT(xR, yR, xCR, yCR);
%     close all,

    [p1, p2]= getRANSAC(rgb2gray(im), rgb2gray(imR));
    nPointsR = size(p1, 1);
    hR = getDLT(p1(:,1), p2(:,1), p1(:,2), p2(:,2),nPointsR);

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
