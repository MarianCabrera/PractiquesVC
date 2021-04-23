clearvars,
close all,
clc,

im = imread(strcat("im1.jpg"));
% imshow(im)

K=2;

c = double(randomClusters(K, "RGB"));

rgbVector = double(reshape(im, [], 3));
done = 0;
loop = 0;
while done == 0
    loop = loop + 1
    I = assignClusters(c, rgbVector);
    [c, done] = recalculateClusters(c, rgbVector, I)
end



function c = randomClusters(K, type)
    switch(type)
        case "RGB"
            c = randi([0, 255], K, 3);
        case "HSV"
            c = rand(K, 3);
    end
end

function I = assignClusters(c, v)
    dist = zeros(size(c, 1), size(v, 1));
    for i = 1 : size(c, 1)
        dist(i, :) = abs(sum(c(i,:) - v(:,:), 2));
    end
    [M, I] = min(dist, [], 1);
end

function [newC, done] = recalculateClusters(c, v, I)
    newC = c;
    done = 0;
    count = zeros(size(c, 1));
    for i = 1 : size(c, 1)
        for j = 1 : size(I,2)
            if I(j) == i
                newC(i, :) = newC(i, :) + v(j, :);
                count(i) = count(i) + 1;
            end
        end
        newC(i, :) = newC(i, :) / count(i);
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
                newC(i, :) = newC(i, :) + v(j, :);
                count(i) = count(i) + 1;
            end
        end
    end


end

