path = "img/cnm/";
left = imread(strcat(path, "image011.jpg"));
center = imread(strcat(path, "image012.jpg"));
right = imread(strcat(path, "image013.jpg"));

left = double(left)/255;
center = double(center)/255;
right = double(right)/255;

imOut =  getImage(left, center, right, 4);
imshow(imOut)

function [x, y] = getPointsImage(img, nPoints, n)
    figure(n),imshow(img)
    [x, y] = ginput(nPoints);
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

function c = getCornersToH(im, h, nPoints)
    s = size(im);
    corners = [0 0; s(2) 0; s(2) s(1); 0 s(1)];
    c = zeros(nPoints,2);
    for i = 1: nPoints
        p = applyDLT(corners(i, 1), corners(i, 2), h);
        c(i, 1) = p(1);
        c(i, 2) = p(2);
    end
end

function imOut = getImage(imL, im, imR, nPoints)
    [xL, yL] = getPointsImage(imL, nPoints, 1);
    [xCL, yCL] = getPointsImage(im, nPoints, 2);
    hL = getDLT(xL, yL, xCL, yCL);
    [xR, yR] = getPointsImage(imR, nPoints, 3);
    [xCR, yCR] = getPointsImage(im, nPoints, 4);
    hR = getDLT(xR, yR, xCR, yCR);
    close all,

    cL = getCornersToH(imL, hL, nPoints);
    cR = getCornersToH(imR, hR, nPoints);

    c = [cL; cR];
    
    minX = min(c(:,1));
    minY = min(c(:,2));
    maxX = max(c(:,1));
    maxY = max(c(:,2));
    
    sizeY = maxY - minY;
    sizeX = maxX - minX;
    
    imOut = double(zeros(int32(sizeY), int32(sizeX), 3));
    imOutPlus = double(zeros(int32(sizeY), int32(sizeX)));
    
    invHL = inv(hL);
    invHR = inv(hR);

end




