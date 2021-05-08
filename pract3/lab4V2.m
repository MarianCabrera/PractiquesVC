function lab4
    close all,

    path = "img/cnm/";
    left = imread(strcat(path, "image010.jpg"));
    center = imread(strcat(path, "image012.jpg"));
    right = imread(strcat(path, "image014.jpg"));
    
    left2 = imread(strcat(path, "image008.jpg"));
    right2 = imread(strcat(path, "image016.jpg"));
    
    left3 = imread(strcat(path, "image006.jpg"));
    right3 = imread(strcat(path, "image018.jpg"));
    
    images{1} = left3;
    images{2} = left2;
    images{3} = left;
    images{4} = center;
    images{5} = right;
    images{6} = right2;
    images{7} = right3;
    

%     path = "img/meves/";
%     left = imread(strcat(path, "3.jpg"));
%     center = imread(strcat(path, "2.jpg"));
%     right = imread(strcat(path, "1.jpg"));

%     left = double(left)/255;
%     center = double(center)/255;
%     right = double(right)/255;

    

    nPoints = 4;
    imOut = getImage(left, center, right, nPoints);
    figure(6)
    imshow(uint8(imOut))
    imOut = getImage(left2, imOut, right2, nPoints);
    figure(7)
    imshow(uint8(imOut))
%     imOut3 = getImage(left3, imOut2, right3, nPoints);
%     figure(8)
%     imshow(uint8(imOut3))
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

function imOut = getImage(images, nPoints)
    type = 1;
    for i = 1:size(images,2)
        imcil{i} = imToCil(images{i});
    end
%     im = imToCil(im1);
%     imL = imToCil(imL1);
%     imR = imToCil(imR1);
    
    switch type
        case 0
%             [xL, yL] = getPointsImage(imL, nPoints, 1);
%             [xCL, yCL] = getPointsImage(im, nPoints, 2);
%             [xR, yR] = getPointsImage(imR, nPoints, 3);
%             [xCR, yCR] = getPointsImage(im, nPoints, 4);
%             close all,
%             hL = getCilH(xL, yL, xCL, yCL, nPoints);
%             hR = getCilH(xR, yR, xCR, yCR, nPoints);
        case 1
            for i = 2 : size(images,2)-1
                [xL, yL, xCL, yCL] = getSURF(imcil{i-1}, imcil{i});
                [xCR, yCR, xR, yR] = getSURF(imcil{i+1},  imcil{i});
                sizeL = size(xL, 1);
                sizeR = size(xR, 1);
                hL{i} = getCilH(xL, yL, xCL, yCL, sizeL);
                hR{i} = getCilH(xR, yR, xCR, yCR, sizeR);
            end
%             [xL, yL, xCL, yCL] = getSURF(imL, im);
%             [xCR, yCR, xR, yR] = getSURF(im, imR);
%             sizeL = size(xL, 1);
%             sizeR = size(xR, 1);
%             hL = getCilH(xL, yL, xCL, yCL, sizeL);
%             hR = getCilH(xR, yR, xCR, yCR, sizeR);
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