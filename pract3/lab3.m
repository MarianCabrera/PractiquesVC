
path = "img/cnm/";
left = imread(strcat(path, "image011.jpg"));
center = imread(strcat(path, "image012.jpg"));
right = imread(strcat(path, "image013.jpg"));

imOut = generateLargeImg(left, center, right, 4);
figure(5), imshow(uint8(imOut))

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

function imOut = generateLargeImg(imL, im, imR, nPoints)
% predef_xL = [1562 2243 2219 1568];
% predef_yL = [1163 1256 710 716];
% predef_xCL = [641 1304 1289 650];
% predef_yCL = [1160 1265 722 713];
% predef_xR = [233 926 935 223];
% predef_yR = [1277 1430 791 707];
% predef_xCR = [1307 1979 1973 1286];
% predef_yCR = [1277 1424 779 716];

    [xL, yL] = getPointsImage(imL, nPoints, 1);
    [xCL, yCL] = getPointsImage(im, nPoints, 2);
    hL = getDLT(xL, yL, xCL, yCL);
%     hL = getDLT(predef_xL, predef_yL, predef_xCL, predef_yCL);

%     [xR, yR] = getPointsImage(imR, nPoints, 3)
%     [xCR, yCR] = getPointsImage(im, nPoints, 4)
%     hR = getDLT(xR, yR, xCR, yCR);
%     hR = getDLT(predef_xR, predef_yR, predef_xCR, predef_yCR);
    close all,
    
    s = size(im);
    
    sL = size(imL);
    cornersL = [0 0; sL(2) 0; sL(2) sL(1); 0 sL(1)];
    cornersLtoH = zeros(4,2);
    for i = 1: nPoints
        p = applyDLT(cornersL(i, 1), cornersL(i, 2), hL);
        cornersLtoH(i, 1) = p(1);
        cornersLtoH(i, 2) = p(2);
    end
    
    sR = size(imR);
    cornersR = [0 0; sR(2) 0; sR(2) sR(1); 0 sR(1)];
    cornersRtoH = zeros(4,2);
    for i = 1: nPoints
        p = applyDLT(cornersR(i, 1), cornersR(i, 2), hR);
        cornersRtoH(i, 1) = p(1);
        cornersRtoH(i, 2) = p(2);
    end
    
    cornersToH = [cornersLtoH; cornersRtoH];
    
    minX = min(cornersToH(:,1));
    minY = min(cornersToH(:,2));
    maxX = max(cornersToH(:,1));
    maxY = max(cornersToH(:,2));
    
    sizeY = maxY - minY;
    sizeX = maxX - minX;
    imOut = zeros(int32(sizeY), int32(sizeX), 3);
    imOutPlus = zeros(int32(sizeY), int32(sizeX));
    imOut(int32(0-minY+1): int32(s(1)+(0-minY)), int32(0-minX+1): int32(s(2)+(0-minX)), :) = im;
    
    invHL = inv(hL);
    invHR = inv(hR);

    for i = 1: size(imOut, 1)
        for j = 1 : size(imOut, 2)
            realX = i + minX;
            realY = j + minY;
            p = applyDLT(realX, realY, invHL);
            if (p(1) > 0 && p(1) < sL(1)-1 && p(2) > 0 && p(2) < sL(2)-1)
                indexX = int32(p(1))+1;
                indexY = int32(p(2))+1;

%               TODO passar a double / 255
%                 imOut(i, j, :) = imOut(i, j, 1) + imL(indexX, indexY, 1);
                imOut(i, j, :) = imL(indexX, indexY, :);
%                 imOutPlus(i, j) = imOutPlus(i, j)+1;
%                     imOut(j, i, :) =  imL(indexY+1, indexX+1, :);
%                     imOut(j, i, :) = imOut(j, i, :) + imL(indexY+1, indexX+1, :);
                    
               
            end
        end
    end
    
%     for i = 1: size(imOut, 1)
%         for j = 1 : size(imOut, 2)
%             if imOutPlus(i,j) > 1
%                  imOut(i, j, :) = imOut(i, j, :)/imOutPlus(i,j);
%             end
%         end
%     end

end
















