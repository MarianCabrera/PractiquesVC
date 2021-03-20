% INIT 
clearvars,
close all,
clc,

im = imread('sonw.jpg');
result = zeros(size(im));
size1 = size(result);

result(:,:,1) = im(:,:,1);
result(:,:,2) = im(:,:,2);
result(:,:,3) = im(:,:,3);

imshow(uint8(result))

padd1 = zeros(size1(1)+20, size1(2)+20);
padd1(1:(size1(1)-20), 1:(size1(2)-20)) = im(:,:,1);
imshow(padd1)

padd2 = zeros(size1(1)+20, size1(2)+20);
padd3 = zeros(size1(1)+20, size1(2)+20);



