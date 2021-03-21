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

result2 = align(im(:,:,1),im(:,:,2),im(:,:,3),0);
%imshow(uint8(result2))
 
% function result = test(im1, im2, im3)
%     s = size(im1);
%     result = zeros(s(1), s(2) ,3);
%     result(:,:,1) = im1;
%     result(:,:,2) = im2;
%     result(:,:,3) = im3;
% end

function result = fftConv(ref, im)
    result = ifft2(fft2(ref).*conj(fft2(im)));
    imshow(result)
end

function aligned = align(ref, im2, im3, mode)
    s = size(ref);
    aligned = zeros(s(1), s(2), 3);
    if mode == 0
       aligned(:,:,1) = ref;
       aligned(:,:,2) = fftConv(ref, im2);
       aligned(:,:,3) = fftConv(ref, im3);
    end
end
% padd1 = zeros(size1(1)+20, size1(2)+20);
% padd1(1:(size1(1)-20), 1:(size1(2)-20)) = im(:,:,1);
% imshow(padd1)
% 
% padd2 = zeros(size1(1)+20, size1(2)+20);
% padd3 = zeros(size1(1)+20, size1(2)+20);



