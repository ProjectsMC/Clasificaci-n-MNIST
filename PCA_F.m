function [train_PCA,test_PCA,outmse,W]=PCA_F(x,train_datan,test_datan)
% load ('Trainnumbers.mat'); 
ncompca=x;

[D,N]=size(train_datan);

clear in meani stdi;
% meani=mean(train_data')';
% stdi=std(train_data')';

%Elimina los datos que tienen varianza 0
% train_data_w0=train_data;
% stdi_w0=stdi;
% meani_w0=meani;

% for i=1:D
%     if stdi(i)==0
%         train_data_w0(:,i)=[];
%     end
% end
[D,N]=size(train_datan); %Nuevas dimensiones
%meani=mean(train_data')';
%stdi=std(train_data')';
% % stdi_w0(stdi_w0==0)=0.01;
% for i=1:N
%     in(:,i)=(train_data(:,i)-meani)./stdi;
%     for j=1:D
%         if isnan(in(j,i))
%             in(j,i)=0;
%         end
%     end
%     % div(isinf(div)) = 0;
%     % in(:,i)=div;
% end
% 
% %Probando 
% 
% % in_no_zeros = in(any(in, 2), :);
% % covariance_matrix = cov(in_no_zeros');
% % 
[Wc,Diag]=eig(cov(train_datan'));
% [New_D,~]=size(Wc)

 W=zeros(ncompca,D);
 diagonal=diag(Diag);
 sum_diag_tot=sum(Diag(:));

 %Ordenando la matriz acorde a los valores de la diagonal
 [valores_ordenados, indices_ordenados] = sort(diagonal,'ascend');
 New_Wc=Wc(indices_ordenados,:);
 sum_diag=0;
for i=1:ncompca
    W(i,:)=New_Wc(:,D+1-i)';
    sum_diag=sum_diag+Diag((D+1-i),(D+1-i));
end
 inproj=W*train_datan;
 inrecons=W' * inproj;
 ExpectedError=sum_diag_tot-sum_diag; 

 train_PCA=inproj;
 test_PCA=W*test_datan;


%Denormalized data
% meanp=mean(p.value')'; stdp=std(p.value')';
% projected_data_denormalized=zeros(D,N);
% for i=1:N
%     projected_data_denormalized(:,i)= inrecons(:,i).*stdi+meani;
% 
% end
% 
% % %Calculo del MSE
% mse_reconstruction_expected = ExpectedError;
% mse_reconstruction_actual=0;
% mse_original_data=0;
% for i=1:N
%     mse_reconstruction_actual=mse_reconstruction_actual+ (norm(inrecons(:,i)-train_datan(:,i))^2);
% end
% for i=1:N
%     mse_original_data=mse_original_data+ (norm(projected_data_denormalized(:,i)-train_datan(:,i))^2);
% end
% outmse=mse_reconstruction_actual/N;
outmse=1;
% % Paso 12: Muestra los resultados
% fprintf('Expected MSE of normalized data: %f\n', mse_reconstruction_expected);
% fprintf('Actual MSE of the normalized data: %f\n', mse_reconstruction_actual/N);
% fprintf('Actual MSE of the not normalized data: %f\n', mse_original_data/N);
