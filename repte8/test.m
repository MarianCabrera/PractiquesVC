clearvars,
close all,
clc,

load fisheriris
X = meas;
Y = species;
rng('default')  % for reproducibility

% % 8.1 Reducció de la dimensió
% col=hsv2rgb([(0:9)/10; ones(2,10)]'); %colors per representar les dades
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % dades sintètiques (2 classes)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% n = 10000;
% d = 4;
% X = randn(d,n);
% M = rand(d,d)/(d/4);
% M1 = [M, rand(d,1)];
% M2 = [M, rand(d,1)];
% X = [M1*[X(:,1:end/2); ones(1,n/2)], M2*[X(:,end/2+1:end); ones(1,n/2)]];
% figure,
% plot(X(1,1:end/2),X(2,1:end/2),'.','Color',col(1,:)), hold on
% plot(X(1,end/2+1:end),X(2,end/2+1:end),'.','Color',col(5,:))
% axis equal
% title ('primeres 2 dimensions')