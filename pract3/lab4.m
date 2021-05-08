function lab4
    close all,

    path = "img/ee/";
    left = imread(strcat(path, "image005.jpg"));
    center = imread(strcat(path, "image007.jpg"));
    right = imread(strcat(path, "image009.jpg"));
    
    left2 = imread(strcat(path, "image003.jpg"));
    right2 = imread(strcat(path, "image011.jpg"));
    
    left3 = imread(strcat(path, "image001.jpg"));
    right3 = imread(strcat(path, "image013.jpg"));

%     path = "img/meves/";
%     left = imread(strcat(path, "3.jpg"));
%     center = imread(strcat(path, "2.jpg"));
%     right = imread(strcat(path, "1.jpg"));

    nPoints = 4;
    imOut = getImage(imToCil(left), imToCil(center), imToCil(right), nPoints);
    figure(6)
    imshow(uint8(imOut))
    imOut = getImage(imToCil(left2), imToCil(imOut), imToCil(right2), nPoints);
    figure(7)
    imshow(uint8(imOut))
%     imOut3 = getImage(left3, imOut2, right3, nPoints);
%     figure(8)
%     imshow(uint8(imOut3))
end

function [x1, y1] = getPointsImage(img, nPoints, n)
    figure(n),imshow(img)
    enableDefaultInteractivity(gca);
    x1=[]; y1=[];
    for j=1:nPoints
        [x1(j),y1(j)]=ginput(1);
        figure(n)
        hold on
        plot(x1(j),y1(j),'o','MarkerSize',10,'LineWidth',10);
        text(x1(j),y1(j),string(j));
        hold off
    end
end

function [x1, y1, x2, y2] = getSURF(im1, im2)
    im1 = rgb2gray(im1);
    im2 = rgb2gray(im2);
    
     points1 = detectSURFFeatures(im1);
     points2 = detectSURFFeatures(im2);
 
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

function H = getCilH(x1, y1, x2, y2, points)
    totalX = 0;
    totalY = 0;
    for i=1:points
        totalX = totalX + (x2(i) - x1(i));
        totalY = totalY + (y2(i) - y1(i));
    end
    totalX = totalX / points;
    totalY = totalY / points;
    
    H = [1 , 0, totalX; 0, 1, totalY; 0, 0, 1]
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

function [xNew, yNew] = plaToCilindre(x, y)
    s = double(2000); f = s;
    xNew = s * atan(double(x) / f);
    yNew = s * (double(y) / (sqrt((double(x)^2) + (f^2))));
end

function [xNew, yNew] = cilindreToPla(x, y)
    s = double(2000); f = s;
    xNew = f * tan(double(x) / s);
    yNew = f * (double(y) / s) * sec(double(x) / s);
end

function imROutIn = imToCil(im)
    centerX =( size(im, 2)-1) /2;
    centerY = (size(im, 1)-1) /2;
    cornersCentered = [1-centerX, 1-centerY; size(im, 2)-centerX, 1-centerY; size(im, 2)-centerX, size(im, 1)-centerY; 1-centerX, size(im, 1)-centerY];
    newCornersCentered = zeros(4, 2);

    for i = 1 : 4
        [newX, newY] = plaToCilindre(cornersCentered(i,1), cornersCentered(i, 2));
        newCornersCentered(i, 1) = newX;
        newCornersCentered(i, 2) = newY;
    end

    xplimCentered = [floor(min(newCornersCentered(:,1))), ceil(max(newCornersCentered(:,1)))];
    yplimCentered = [floor(min(newCornersCentered(:,2))), ceil(max(newCornersCentered(:,2)))];

    xplim = xplimCentered + centerX;
    yplim = yplimCentered + centerY;

    imROutIn = uint8(zeros(yplim(2) - yplim(1)-1, xplim(2) - xplim(1)-1, 3));

    newCenterX = size(imROutIn, 2) /2;
    newCenterY = size(imROutIn, 1) /2;

    for i = 1:size(imROutIn, 1)
        for j = 1:size(imROutIn, 2)
            xpCenter = j - newCenterX+1;
            ypCenter = i - newCenterY+1;

            [xCenter, yCenter] = cilindreToPla(xpCenter, ypCenter);
            x = int32(xCenter + centerX);
            y = int32(yCenter + centerY);
            if(x > 0 && x <= size(im,2) && y > 0 && y <= size(im,1))
                imROutIn(i,j,:) = im(y,x,:);
            end

        end
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
            hL = getCilH(xL, yL, xCL, yCL, nPoints);
            hR = getCilH(xR, yR, xCR, yCR, nPoints);
        case 1
            [xL, yL, xCL, yCL] = getSURF(imL, im);
            [xCR, yCR, xR, yR] = getSURF(im, imR);
            sizeL = size(xL, 1);
            sizeR = size(xR, 1);
            hL = getCilH(xL, yL, xCL, yCL, sizeL);
            hR = getCilH(xR, yR, xCR, yCR, sizeR);
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
    
    imOut = double(zeros(int32(sizeY)-1, int32(sizeX)-1, 3));
    imOutPlus = double(zeros(int32(sizeY)-1, int32(sizeX)-1));
    imOut(int32((0 - minY) +1) : int32(size(im, 1) + (0 - minY)), int32((0 - minX)+1) : int32(size(im, 2) + (0 - minX)), :) = im(:,:,:);
    imOutPlus(int32((0 - minY) +1) : int32(size(im, 1) + (0 - minY)), int32((0 - minX)+1) : int32(size(im, 2) + (0 - minX))) = 1;
    
    invHL = inv(hL);
    invHR = inv(hR);
    
    for j = 1 : size(imOut, 2)
        for i = 1 : size(imOut, 1)
            xToH = j + minX+1;
            yToH = i + minY+1;
            pL = applyDLT(xToH, yToH, invHL);
            pX = int32(pL(1));
            pY = int32(pL(2));
            if pX>0 && pX<size(imL, 2) && pY>0 && pY<size(imL, 1)
                imOut(i, j, :) = double(imOut(i, j, :)) + double(imL(pY, pX, :));
                imOutPlus(i, j) =  imOutPlus(i, j) + 1;
            end
            pR = applyDLT(xToH, yToH, invHR);
            pX = int32(pR(1));
            pY = int32(pR(2));
            if pX>0 && pX<size(imR, 2) && pY>0 && pY<size(imR, 1)
                imOut(i, j, :) = double(imOut(i, j, :)) + double(imR(pY, pX, :));
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
