clearvars,
close all,
clc,

sig = 16;
k = 0.05;

im1 = imread('i1.jpg');
% imshow(uint8(im1))
im2 = imread('i2.jpg');
% imshow(uint8(im2))

points1 = getHarrisPoints(double(rgb2gray(im1)), sig, k);
points2 = getHarrisPoints(double(rgb2gray(im2)), sig, k);

figure(1),
imshow(im1,[])
hold on;
plot(points1(1,:),points1(2, :),'ro','MarkerSize',2*sig);
drawnow
hold off;

figure(2),
imshow(im2,[])
hold on;
plot(points2(1,:),points2(2, :),'ro','MarkerSize',2*sig);
drawnow
hold off;

desc1 = setDescriptors(im1, points1);
desc2 = setDescriptors(im2, points2);

[m1, m2] = setMatch(desc1, desc2, points1, points2);

plotColor={'wo','ro','yo','go','co','bo','mo','ko'};
figure,
imshow([im1,im2]);
hold on
despl = size(im1,2);
plot(points1(1,:),points1(2, :),plotColor{1},'MarkerSize',5);
plot(points2(1,:)+despl,points2(2, :),plotColor{2},'MarkerSize',5);
for j=1:500
    line([m1(1,j),m2(1,j)+despl],[m1(2,j), m2(2,j)]);
end
hold off

function points = getHarrisPoints(im, sig, k)
    Ix = conv2(im, [-1, 0, 1], 'same');
    Iy = conv2(im, [-1, 0, 1]', 'same');
    g = fspecial('gaussian',max(1,fix(6*sig)),sig);

    Ag = conv2(Ix .^2, g, 'same');
    Bg = conv2(Iy .^2, g, 'same');
    Cg = conv2(Ix .* Iy, g, 'same');

    R = (Ag.*Bg - Cg.^2) - k*(Ag + Bg).^2;
    
    p = imregionalmax((R>1000).*R);
    [row,col] = find(p);
    val = R(sub2ind(size(R),row,col));

    N = 100; 
    N = min(N,length(val));
    [valsort,ind]=sort(val,1,'descend');
    
    points = zeros(2, N);
    for i = 1 : N
        points(1, i) = col(ind(i));
        points(2, i) = row(ind(i));
    end 
end

function desc = setDescriptors(im, points)
    desc = zeros(size(points,2),75);
    for i = 1 : size(points,2)
        xCenter = points(1,i);
        yCenter = points(2,i);
        vals = im(yCenter - 2 : yCenter + 2, xCenter - 2 : xCenter + 2, :);
        desc(i, :) = reshape(vals,1,[]);
    end
end

function [m1, m2] = setMatch(d1, d2, p1, p2)
     for j = 1 : size(p1,2)
        d1Val = d1(j, :);
        [~, match]= min(sum(abs(d2 - d1Val)'));
        
        m1(:, j) = p1(:, j);
        m2(:, j) = p2(:, match);   
     end
end