% INIT 
% clearvars,
% close all,
% clc,



path = "img/cnm/";
left = imread(strcat(path, "image011.jpg"));
center = imread(strcat(path, "image012.jpg"));
right = imread(strcat(path, "image013.jpg"));
figure(1),imshow(left)
[x1 y1] = ginput(4);

figure(2),imshow(center)
[x2 y2] = ginput(4);

% H [x1 y1 1]' = [x2 y2 1]'
% fico H com un vector h=[h11 ... h33]' i busco les equacions M*h=0
% DLT (Direct Linear Transformation, sense normalitzar)
M = [];
for i=1:4
    M = [ M ;
    x1(i) y1(i) 1 0 0 0 -x2(i)*x1(i) -x2(i)*y1(i) -x2(i);
    0 0 0 x1(i) y1(i) 1 -y2(i)*x1(i) -y2(i)*y1(i) -y2(i)];
end
[u,s,v] = svd( M );
H = reshape( v(:,end), 3, 3 )';
H = H / H(3,3);
imSize = size(left)
leftCenter = zeros(imSize(1), imSize(2)*2);

% fi DLT
b = 1;
while(b ~= 3)
    figure(1)
    [x y b] = ginput(1);
    p = H*[x y 1]';
    p = p/p(3);
    figure(2)
    hold on
    plot(2500,1900,'rx','MarkerSize',20,'LineWidth',3);
    plot(p(1),p(2),'rx','MarkerSize',20,'LineWidth',3);
    hold off
end

