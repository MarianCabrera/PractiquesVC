% clearvars,
% close all,
% clc,

path = "img/cnm/";
left = imread(strcat(path, "image011.jpg"));
im = imread(strcat(path, "image012.jpg"));
right = imread(strcat(path, "image013.jpg"));

left = double(left)/255;
% im = double(im)/255;
right = double(right)/255;

centerX = size(im, 2) /2;
centerY = size(im, 1) /2;
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

imROutIn = uint8(zeros(yplim(2) - yplim(1), xplim(2) - xplim(1), 3));

newCenterX = size(imROutIn, 2) /2;
newCenterY = size(imROutIn, 1) /2;

for i = 1:size(imROutIn, 1)
    for j = 1:size(imROutIn, 2)
        xpCenter = j - newCenterX;
        ypCenter = i - newCenterY;
        
        [xCenter, yCenter] = cilindreToPla(xpCenter, ypCenter);
        x = int32(xCenter + centerX);
        y = int32(yCenter + centerY);
        if(x > 0 && x <= size(im,2) && y > 0 && y <= size(im,1))
            imROutIn(i,j,:) = im(y,x,:);
        end
        
    end
end

imshow(imROutIn)

function [xNew, yNew] = plaToCilindre(x, y)
    s = double(1000); f = s;
    xNew = s * atan(double(x) / f);
    yNew = s * (double(y) / (sqrt((double(x)^2) + (f^2))));
end

function [xNew, yNew] = cilindreToPla(x, y)
    s = double(1000); f = s;
    xNew = f * tan(double(x) / s);
    yNew = f * (double(y) / s) * sec(double(x) / s);
end