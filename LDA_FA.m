function [train_LDA,test_LDA,w_lda]=LDA_FA(train_data,test_data,Number_groups)

%LDA Adaptativo

% cluster_per_class=[1,2,2,1,3,2,2,2,2,3];
% 
% [D,N]=size(train_data.images);
% number_per_class=N/10;
% total_sub_classes=sum(cluster_per_class);
% new_labels=zeros(1,total_sub_classes);
% for i=1:10
%     idx = kmeans(train_data.images_n(:,(number_per_class*(i-1)+1:(number_per_class*i+1)))', cluster_per_class(i));
%     new_labels((number_per_class*(i-1)+1:(number_per_class*i+1)))=idx+sum(cluster_per_class(1:(i-1)));
% end
% 


[D,N]=size(train_data.im_PCA);
meani=mean(train_data.im_PCA')';
number_per_class=zeros(1,Number_groups);
for a=1:N
    number_per_class(1,(train_data.kmeans_labels(1,a)))=number_per_class(1,(train_data.kmeans_labels(1,a)))+1;
end
SW=zeros(D);
SB=[];
global weights;
weights=number_per_class
sum_c=zeros(D,Number_groups);
for m=1:N
    % if train_data.kmeanslabels(1,m)==0
    %     j=10;
    % else
    j=train_data.kmeans_labels(1,m);
    % end
sum_c(:,j)=sum_c(:,j)+train_data.im_PCA(:,m);
end
for k=1:Number_groups
mean_c(:,k)=sum_c(:,k)/number_per_class(1,k);
end

for h=1:N
    % if train_data.labels(1,h)==0
    %     c=10;
    % else
    c=train_data.kmeans_labels(1,h);
    % end
SW=SW+((train_data.im_PCA(:,h)-mean_c(:,c))*(train_data.im_PCA(:,h)-mean_c(:,c))');
end
% [S1,S2]=SC(pn,m1,m2)
% SW=S1+S2
SB=number_per_class(1,1)*(mean_c(:,1)-meani)*(mean_c(:,1)-meani)';
for q=2:Number_groups
    SB=SB+(number_per_class(1,q)*(mean_c(:,q)-meani)*(mean_c(:,q)-meani)');
end

[Wlda,Diag]=eig(pinv(SW)*SB);
 sum_diag=0;

 diagonal=diag(Diag);
 sum_diag_tot=sum(Diag(:));

 %Ordenando la matriz acorde a los valores de la diagonal
 [valores_ordenados, indices_ordenados] = sort(diagonal,'ascend');
 New_wlda=Wlda(:,indices_ordenados);

for i=1:Number_groups-1
    Wf(i,:)=New_wlda(:,D+1-i)';
    sum_diag=sum_diag+valores_ordenados(D+1-i);
end
% Wf(1,:)=Wlda(:,D)';
% Wpca(1,:)=Wc(:,D)';

% pnproj1=Wpca*tn.value;
train_LDA=Wf*train_data.im_PCA;
test_LDA=Wf*test_data.im_PCA;
w_lda=Wf;
end
% Coord1(1,:)=pnproj1(1,:);
% Coord2(1,:)=pnproj2(1,:);
% pnrecons1=Wpca'*Coord1;
% pnrecons2=Wf'*Coord2;
% ExpectedError=Diag(1,1)