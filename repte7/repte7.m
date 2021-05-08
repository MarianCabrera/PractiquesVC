clearvars,
close all,
clc,

% im = imread(strcat("im1.jpg"));
im = imread(strcat("t1.png"));
% imshow(im)

K=16;

c = double(randomClusters(K, "HSV"));

rgbVector = double(reshape(im,size(im,1)*size(im,2),3));
hsvVector = double(reshape(rgb2hsv(im),size(im,1)*size(im,2),3));
done = 0;
loop = 0;


while done == 0 && loop < 50
    loop = loop + 1
    [I,dist] = assignClusters(c, rgbVector);
    [c, done] = recalculateClusters(c, rgbVector, I);
end

imNew = reasignColors(im, c, rgbVector, I);
% imshow(hsv2rgb(imNew))
imshow(uint8(imNew))

function c = randomClusters(K, type)
    switch(type)
        case "RGB"
            c = randi([0, 255], K, 3);
        case "HSV"
            c = rand(K, 3);
    end
end

function [I,dist] = assignClusters(c, v)
    dist = zeros(size(c, 1), size(v, 1));
    for i = 1 : size(c, 1)
        dist(i, :) = sum(c(i,:) - v(:,:), 2) .* sum(c(i,:) - v(:,:), 2) ;
    end
    [M, I] = min(dist, [], 1);
end

function [newC, done] = recalculateClusters(c, v, I)
    newC = c;
    done = 0;
    count = zeros(1,size(c, 1));
    for i = 1 : size(c, 1)
        for j = 1 : size(I,2)
            if I(j) == i
                newC(i, :) = newC(i, :) + v(j, :);
                count(1,i) = count(1,i) + 1;
            end
        end
        newC(i, :) = newC(i, :) / count(1,i);
    end
    
    if newC == c
        done = 1;
        
    else
        done = 0;
    end
    
end

function imNew = reasignColors(im, c, v, I)
    imNew = v;
    
    for i = 1 : size(c, 1)
        for j = 1 : size(I,2)
            if I(j) == i
                imNew(j, :) = c(i, :);
            end
        end
    end
    imNew = reshape(imNew,size(im));
    
end

