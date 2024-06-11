% % Aca se divide de entrada los datos de training y test
% frac= input( 'Ingrese la fraccion que se usara para training: \n' );
% [train_indices,test_indices]=get_percentage(frac,Trainnumbers);
% train_data.images=(Trainnumbers.image(:,train_indices));
% train_data.labels=(Trainnumbers.label(train_indices));
% test_data.images=(Trainnumbers.image(:,test_indices));
% test_data.labels=(Trainnumbers.label(test_indices));
% 
% % Aca normalizamos las imagenes y se adjuntan a la base de datos de cada
% % set
% [train_data.images_n,test_data.images_n]=normalize(train_data.images,test_data.images);

function [train_LDA,test_LDA]=LDA_F(train_data,test_data)


[D,N]=size(train_data.im_PCA);
meani=mean(train_data.im_PCA')';
number_per_class=N/10;
SW=zeros(D);
SB=[];
% ST=zeros(D);
% total1=0;
% total2=0;
% for n=1:N 
%     if p.class(n)==1
%       total1=total1+pn.value(:,n);
%     elseif p.class(n)==2 
%       total2=total2+pn.value(:,n);
%     end
% end
sum_c=zeros(D,10);
for m=1:N
    if train_data.labels(1,m)==0
        j=10;
    else
    j=train_data.labels(1,m);
    end
sum_c(:,j)=sum_c(:,j)+train_data.im_PCA(:,m);
end
mean_c=sum_c/number_per_class;

% for n=1:N
%     if train_data.label(1,n)==1
%     ones=[ones,i];
%         elseif train_data.label(1,n)==2
%         twos=[twos,i];
%             elseif train_data.label(1,n)==3
%             threes=[threes,i];
%             elseif train_data.label(1,n)==4
%             fours=[fours,i];
%                 elseif train_data.label(1,n)==5
%                 fives=[fives,i];
%                     elseif train_data.label(1,n)==6
%                     sixs=[sixs,i];
%                 elseif train_data.label(1,n)==7
%                 sevens=[sevens,i];
%             elseif train_data.label(1,n)==8
%             eights=[eights,i];
%         elseif train_data.label(1,n)==9
%         nines=[nines,i];
%     elseif train_data.label(1,n)==0
%         zeros=[zeros,i];
%     end
% end
% m1=total1/n1;
% m2=total2/n2;
for h=1:N
    if train_data.labels(1,h)==0
        c=10;
    else
    c=train_data.labels(1,h);
    end
SW=SW+((train_data.im_PCA(:,h)-mean_c(:,c))*(train_data.im_PCA(:,h)-mean_c(:,c))');
end
% [S1,S2]=SC(pn,m1,m2)
% SW=S1+S2
SB=number_per_class*(mean_c(:,1)-meani)*(mean_c(:,1)-meani)';
for q=2:10
    SB=SB+(number_per_class*(mean_c(:,q)-meani)*(mean_c(:,q)-meani)');
end

[Wlda,Diag]=eig(pinv(SW)*SB);
 sum_diag=0;

 diagonal=diag(Diag);
 sum_diag_tot=sum(Diag(:));

 %Ordenando la matriz acorde a los valores de la diagonal
 [valores_ordenados, indices_ordenados] = sort(diagonal,'ascend');
 New_wlda=Wlda(:,indices_ordenados);

for i=1:9
    Wf(i,:)=New_wlda(:,D+1-i)';
    sum_diag=sum_diag+valores_ordenados(D+1-i);
end
% Wf(1,:)=Wlda(:,D)';
% Wpca(1,:)=Wc(:,D)';

% pnproj1=Wpca*tn.value;
train_LDA=Wf*train_data.im_PCA;
test_LDA=Wf*test_data.im_PCA;
end
% Coord1(1,:)=pnproj1(1,:);
% Coord2(1,:)=pnproj2(1,:);
% pnrecons1=Wpca'*Coord1;
% pnrecons2=Wf'*Coord2;
% ExpectedError=Diag(1,1)