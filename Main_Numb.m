%MAIN PROGRAM
% % file = input( 'Ingrese el path del file' )
load ('Trainnumbers.mat');
global weights;
weights=[];
% Aca se divide de entrada los datos de training y test
% frac= input( 'Ingrese la fraccion que se usara para training: \n' );
% [train_indices,test_indices]=get_percentage(frac,Trainnumbers);
% train_data.images=(Trainnumbers.image(:,train_indices));
% train_data.labels=(Trainnumbers.label(train_indices));
% test_data.images=(Trainnumbers.image(:,test_indices));
% test_data.labels=(Trainnumbers.label(test_indices));

% Aca normalizamos las imagenes y se adjuntan a la base de datos de cada
% % set
% [train_data.images_n,test_data.images_n]=normalize(train_data,test_data);

%PCA redimensionamiento
x = input( 'Ingrese la dimension a la que busca reducir:' ); %Dimension a reducir
[train_data.im_PCA,test_data.im_PCA,outmse]=PCA_F(x,train_data.images_n,test_data.images_n);
% [train_data.im_PCAf,test_data.im_PCAf]=PCA_ff(train_data.images_n,test_data.images_n,x);

%LDA redimensionamiento
[train_data.im_LDA,test_data.im_LDA]=LDA_F(train_data,test_data);

%K-means definido por clases (cluster_per_class=[1,2,2,1,3,2,2,2,2,3],[1,2,2,1,2,2,2,2,2,2])
cluster_per_class=[1,2,1,2,1,2,1,2,2,3];
[train_data.kmeans_labels,groups]=kmeans_f(train_data,cluster_per_class);

%LDA con Kmeans
[train_data.im_LDA_Km,test_data.im_LDA_Km]=LDA_FA(train_data,test_data,groups);


% Verificar el tamaño de X_train
[~,N] = size(train_data.images);

fprintf('La nueva dimension con PCA es:')
disp(size(train_data.im_PCA));

for nn=3:2:7
fprintf('\n\nCon %d neighbours knn:\n',nn);

%Utilizando Knn con PCA
[prediction.knn_pca,prediction_tr.knn_pca]=Knn_F(train_data.im_PCA,train_data.labels,test_data.im_PCA,test_data.labels,nn);
[accuracy.PCA,misclassifications.PCA]=evaluation(prediction.knn_pca',test_data.labels);
[accuracy_tr.PCA,misclassifications_tr.PCA]=evaluation(prediction_tr.knn_pca',train_data.labels);
fprintf('\nLa precisión test del clasificador k-NN con PCA (%d) es: %.2f%%\n',x, accuracy.PCA * 100);
fprintf('Misclassifications on Test dataset para Knn model: %d\n', misclassifications.PCA);
fprintf('La precisión train del clasificador k-NN con PCA (%d) es: %.2f%%\n',x, accuracy_tr.PCA * 100);

%Utilizando Knn con PCA Y Kmeans
[prediction.knn_pca_km,prediction_tr.knn_pca_km]=Knn_F(train_data.im_PCA,train_data.kmeans_labels,test_data.im_PCA,test_data.labels,nn);
prediction.knn_pca_kmr=back_to_original(prediction.knn_pca_km,cluster_per_class);
prediction_tr.knn_pca_kmr=back_to_original(prediction_tr.knn_pca_km,cluster_per_class);

[accuracy.PCA_km,misclassifications.PCA_km]=evaluation(prediction.knn_pca_kmr,test_data.labels);
[accuracy_tr.PCA_km,misclassifications_tr.PCA_km]=evaluation(prediction_tr.knn_pca_kmr,train_data.labels);

fprintf('\nLa precisión test del clasificador k-NN con PCA (%d) y Kmeans es: %.2f%%\n',x, accuracy.PCA_km * 100);
fprintf('Misclassifications on Test dataset para Knn model: %d\n', misclassifications.PCA_km);
fprintf('La precisión train del clasificador k-NN con PCA (%d) y Kmeans es: %.2f%%\n',x, accuracy_tr.PCA_km * 100);


% [accuracy_PCAf,misclassifications_test_PCAf]=Knn_F(train_data.im_PCAf,train_data.labels,test_data.im_PCAf,test_data.labels,5);
% fprintf('\nLa precisión del clasificador k-NN con PCAf (%d) es: %.2f%%\n',x, accuracy_PCAf * 100);
% fprintf('Misclassifications on Test dataset para Knn model: %d\n', misclassifications_test_PCAf);

%Utilizando Knn con LDA

[prediction.knn_lda,prediction_tr.knn_lda]=Knn_F(train_data.im_LDA,train_data.labels,test_data.im_LDA,test_data.labels,nn);
[accuracy.LDA,misclassifications.LDA]=evaluation(prediction.knn_lda',test_data.labels);
[accuracy_tr.LDA,misclassifications_tr.LDA]=evaluation(prediction_tr.knn_lda',train_data.labels);
fprintf('\nLa precisión test del clasificador k-NN con LDA es: %.2f%%\n', accuracy.LDA * 100);
fprintf('Misclassifications on Test dataset para Knn model: %d\n', misclassifications.LDA);
fprintf('La precisión train del clasificador k-NN con LDA es: %.2f%%\n', accuracy_tr.LDA * 100);

%Utilizando Knn con LDA y K-means

[prediction.knn_lda_km,prediction_tr.knn_lda_km]=Knn_F(train_data.im_LDA_Km,train_data.kmeans_labels,test_data.im_LDA_Km,test_data.labels,nn);
prediction.knn_lda_kmr=back_to_original(prediction.knn_lda_km,cluster_per_class);
prediction_tr.knn_lda_kmr=back_to_original(prediction_tr.knn_lda_km,cluster_per_class);

[accuracy.LDA_Km,misclassifications.LDA_Km]=evaluation(prediction.knn_lda_kmr,test_data.labels);
[accuracy_tr.LDA_Km,misclassifications_tr.LDA_Km]=evaluation(prediction_tr.knn_lda_kmr,train_data.labels);

fprintf('\nLa precisión test del clasificador k-NN con LDA y Kmeans es: %.2f%%\n', accuracy.LDA_Km * 100);
fprintf('Misclassifications on Test dataset para Knn model: %d\n', misclassifications.LDA_Km);
fprintf('La precisión train del clasificador k-NN con LDA y Kmeans es: %.2f%%\n', accuracy_tr.LDA_Km * 100);

end

%Utilizando Naive con PCA
[prediction.naive_pca,prediction_tr.naive_pca]=Naive_F(train_data.im_PCA,train_data.labels,test_data.im_PCA,test_data.labels);
[accuracy.naive_PCA,misclassifications.naive_PCA]=evaluation(prediction.naive_pca',test_data.labels);
[accuracy_tr.naive_PCA,misclassifications_tr.naive_PCA]=evaluation(prediction_tr.naive_pca',train_data.labels);

fprintf('\nLa precisión test del clasificador Naive con PCA (%d) es: %.2f%%\n',x, accuracy.naive_PCA * 100);
fprintf('Misclassifications on Test dataset Naive con PCA (%d): %d\n',x, misclassifications.naive_PCA);
fprintf('La precisión train del clasificador Naive con PCA (%d) es: %.2f%%\n',x, accuracy_tr.naive_PCA * 100);
% fprintf('Total de datos en el conjunto de pruebas: %d\n', size(X_test, 2));
% [accuracynaive_PCAf,misclassifications_testnaive_PCAf]=Naive_F(train_data.im_PCAf,train_data.labels,test_data.im_PCAf,test_data.labels);
% fprintf('La precisión del clasificador Naive con PCAf (%d) es: %.2f%%\n',x, accuracynaive_PCAf * 100);

%Utilizando Naive con PCA y kmeans
[prediction.naive_pca_km,prediction_tr.naive_pca_km]=Naive_F(train_data.im_PCA,train_data.kmeans_labels,test_data.im_PCA,test_data.labels);prediction.naive_pca_kmr=back_to_original(prediction.naive_pca_km,cluster_per_class);
prediction_tr.naive_pca_kmr=back_to_original(prediction_tr.naive_pca_km,cluster_per_class);
[accuracy.naive_pca_kmr,misclassifications.naive_pca_kmr]=evaluation(prediction.naive_pca_kmr,test_data.labels);
[accuracy_tr.naive_pca_kmr,misclassifications_tr.naive_pca_kmr]=evaluation(prediction_tr.naive_pca_kmr,train_data.labels);

fprintf('\nLa precisión test del clasificador Naive con PCA y kmeans (%d) es: %.2f%%\n',x, accuracy.naive_PCA * 100);
fprintf('Misclassifications on Test dataset Naive con PCA (%d): %d\n',x, misclassifications.naive_PCA);
fprintf('La precisión train del clasificador Naive con PCA y kmeans(%d) es: %.2f%%\n',x, accuracy_tr.naive_PCA * 100);

%Utilizando Naive con LDA
[prediction.naive_lda,prediction_tr.naive_lda]=Naive_F(train_data.im_LDA,train_data.labels,test_data.im_LDA,test_data.labels);
[accuracy.naive_LDA,misclassifications.naive_LDA]=evaluation(prediction.naive_lda',test_data.labels);
[accuracy_tr.naive_LDA,misclassifications_tr.naive_LDA]=evaluation(prediction_tr.naive_lda',train_data.labels);

fprintf('\n\nLa precisión test del clasificador Naive con LDA es: %.2f%%\n', accuracy.naive_LDA * 100);
fprintf('Misclassifications on Test dataset Naive con LDA : %d\n', misclassifications.naive_LDA);
fprintf('La precisión train del clasificador Naive con LDA es: %.2f%%\n', accuracy_tr.naive_LDA * 100);

%Utilizando Naive con LDA y Kmeans

% nb_classifier = fitcnb(train_data.im_LDA_Km', train_data.kmeans_labels', 'Distribution','normal','Weights',weights');
% prediction.naive_lda_km = predict(nb_classifier,test_data.im_LDA_Km');

[prediction.naive_lda_km,prediction_tr.naive_lda_km]=Naive_F(train_data.im_LDA_Km,train_data.kmeans_labels,test_data.im_LDA_Km,test_data.labels);
prediction.naive_lda_kmr=back_to_original(prediction.naive_lda_km,cluster_per_class);
prediction_tr.naive_lda_kmr=back_to_original(prediction_tr.naive_lda_km,cluster_per_class);

[accuracy.naive_LDA_Km,misclassifications.naive_LDA_Km]=evaluation(prediction.naive_lda_kmr,test_data.labels);
[accuracy_tr.naive_LDA_Km,misclassifications_tr.naive_LDA_Km]=evaluation(prediction_tr.naive_lda_kmr,train_data.labels);
fprintf('\nLa precisión test del clasificador Naive con LDA y Kmeans es: %.2f%%\n', accuracy.naive_LDA_Km * 100);
fprintf('Misclassifications on Test dataset Naive con LDA y Kmeans : %d\n', misclassifications.naive_LDA_Km);
fprintf('La precisión train del clasificador Naive con LDA y Kmeans es: %.2f%%\n', accuracy_tr.naive_LDA_Km * 100);

% Mostrar los resultados
% fprintf('Misclassifications on Test dataset Naive con PCA (%d): %d\n',x, misclassifications_testnaive_PCA);
% fprintf('Misclassifications on Test dataset Naive con PCAf (%d): %d\n',x, misclassifications_testnaive_PCAf);
% fprintf('Misclassifications on Test dataset Naive con LDA : %d\n', misclassifications.naive_LDA);
% fprintf('Misclassifications on Test dataset Naive con LDA y Kmeans : %d\n', misclassifications.naive_LDA_Km);