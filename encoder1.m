% Cargar datos
load('Trainnumbers.mat');
global weights;
weights = [];

% Dividir datos en entrenamiento y prueba
frac = input('Ingrese la fraccion que se usara para training: \n');
[train_indices, test_indices] = get_percentage(frac, Trainnumbers);
train_data.images = Trainnumbers.image(:, train_indices);
train_data.labels = Trainnumbers.label(train_indices);
test_data.images = Trainnumbers.image(:, test_indices);
test_data.labels = Trainnumbers.label(test_indices);

% Normalizar las imágenes
[train_data.images_n, test_data.images_n] = normalize(train_data, test_data);

% Reducción de dimensión con PCA
x = input('Ingrese la dimension a la que busca reducir: '); % Dimension a reducir
[train_data.im_PCA, test_data.im_PCA, outmse] = PCA_F(x, train_data.images_n, test_data.images_n);

% Verificar el tamaño de los datos de entrenamiento después de PCA
fprintf('La nueva dimension con PCA es:\n');
disp(size(train_data.im_PCA));

% Agregar autoencoder
hiddenSize = input('Ingrese el tamaño de la capa oculta del autoencoder: \n');

% Crear y entrenar el autoencoder
autoenc = trainAutoencoder(train_data.im_PCA,hiddenSize,...
        'EncoderTransferFunction','satlin',...
        'DecoderTransferFunction','purelin',...
        'L2WeightRegularization',0.01,...
        'SparsityRegularization',4,...
        'SparsityProportion',0.10);

% Obtener las características del autoencoder
train_data.im_AE = encode(autoenc, train_data.im_PCA);
test_data.im_AE = encode(autoenc, test_data.im_PCA);
fprintf('La nueva dimension con el autoencoder es:\n');
disp(size(train_data.im_AE));

xReconstructed = predict(autoenc, train_data.im_PCA);
xReconstructedTest = predict(autoenc, test_data.im_PCA);
% Codificar los nuevos datos
% Xnew = rand(size(test_data.im_PCA, 1), 1); % Ejemplo de nuevos datos (aleatorios)
Xcodec = encode(autoenc, train_data.im_PCA);
XcodecTest = encode(autoenc, test_data.im_PCA);
% Decodificar los datos codificados
Xreconstructed = decode(autoenc, Xcodec);
XreconstructedTest = decode(autoenc, XcodecTest);
%figure;
%plot( test_data.im_PCA,'r.');
%hold on
%plot(xReconstructed,'go')


