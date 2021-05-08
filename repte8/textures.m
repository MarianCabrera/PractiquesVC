% 8.2 Detector de pell
im=imread('33137395544_439c84d545_k.jpg');
figure
imshow(im)
N=50;

% disp('pell');
% [xp,yp]=ginput(N);
% disp('no pell');
% [xn,yn]=ginput(N);
% save('punts','xp','yp','xn','yn');
load('punts');
hold on
plot(xp,yp,'go');
plot(xn,yn,'rx');
hold off

for i=1:N
    datap(i,:)=im(round(yp(i)),round(xp(i)),:);
    ldatap(i) = 1;
    datan(i,:)=im(round(yn(i)),round(xn(i)),:);
    ldatan(i)=0;
end

% 70/30 training/test
idx=randperm(N);
Ntrai=round(0.7*N);
Ntest=N-Ntrai;
d_trai=[datap(idx(1:Ntrai),:);datan(idx(1:Ntrai),:)];
l_trai=[ldatap(idx(1:Ntrai)), ldatan(idx(1:Ntrai))];
d_test=[datap(idx(Ntrai+1:end),:);datan(idx(Ntrai+1:end),:)];
l_test=[ldatap(idx(Ntrai+1:end)), ldatan(idx(Ntrai+1:end))];


% model per kNN
% les dades de training: d_trai,l_trai
%classificació del test
% distancies per kNN
v_trai=double(permute(repmat(d_trai,[1,1,2*Ntest]),[1,3,2]));
v_test=double(permute(repmat(d_test,[1,1,2*Ntrai]),[3,1,2]));
%[kk,ind]=sort(sqrt(sum((v_trai-v_test).^2,3)),1,'ascending');
[kk,ind]=sort(sqrt(sum((v_trai-v_test).^2,3)));

figure, hold on
for k=1:2:15
    claskNN=l_trai(ind(1:k,:)); %k mostres més properes del training
    %distkNN=kk(1:k,:);

    predictionkNN_simple=sum(claskNN,1)>(k/2);
    TP = sum((l_test==1)&(predictionkNN_simple==1));
    TN = sum((l_test==0)&(predictionkNN_simple==0));
    FN = sum((l_test==1)&(predictionkNN_simple==0));
    FP = sum((l_test==0)&(predictionkNN_simple==1));

    ACC=(TP+TN)/(TP+TN+FP+FN);
    ERR = 1-ACC;
    fprintf('Error rate on the test %5.2f%% for k = %d\n',100*ERR,k);

    %probability (nombre de veïns convertit a probabilitat)
    p=sum(claskNN,1)/k;

    %ROC
    roc(1,:)=[1,1];
    i=2;
    for thr=0:0.05:1
        predictionP=p>thr;
        TP = sum((l_test==1)&(predictionP==1));
        TN = sum((l_test==0)&(predictionP==0));
        FN = sum((l_test==1)&(predictionP==0));
        FP = sum((l_test==0)&(predictionP==1));
        FPR=FP/(FP+TN);
        TPR=TP/(TP+FN);
        roc(i,:)=[FPR,TPR];
        i=i+1;
    end
    plot(roc(:,1),roc(:,2),'-','Color',[1-k/15,k/15,1],'DisplayName',sprintf('k=%d',k));
end
title('Corbes ROC, per diferents k del k-NN')
axis equal
axis([0 1 0 1])
legend({})


im=im(1:4:end,1:4:end,:);

Nfinal=size(im,1)*size(im,2);
d_final=reshape(im,Nfinal,size(im,3));
v_final=double(permute(repmat(d_final,[1,1,2*Ntrai]),[3,1,2]));
v_trai=double(permute(repmat(d_trai,[1,1,Nfinal]),[1,3,2]));
[kk,ind]=sort(sqrt(sum((v_trai-v_final).^2,3)));

k=5;
claskNN=l_trai(ind(1:k,:)); %k mostres més properes del training
p=sum(claskNN,1)/k;
thr=0.7;
res=p>thr;
res=reshape(res,size(im,1),size(im,2));

figure, 
subplot(1,2,1)
imshow(im), title ('imatge original')
subplot(1,2,2)
imshow(res), title ('segmentació amb k-NN,  k=5, p=0.7')



% 8.3 OCR
pathname='./ocr-data/';
l_test=readidx([pathname 't10k-labels.idx1-ubyte']);
i_test=readidx([pathname 't10k-images.idx3-ubyte']);
l_trai=readidx([pathname 'train-labels.idx1-ubyte']);
i_trai=readidx([pathname 'train-images.idx3-ubyte']);

v_trai=reshape(i_trai,size(i_trai,1)*size(i_trai,2),size(i_trai,3))/255;
v_test=reshape(i_test,size(i_test,1)*size(i_test,2),size(i_test,3))/255;

