% Cargar datos
load('Trainnumbers.mat');
% Dividir datos en entrenamiento y prueba
frac = input('Ingrese la fraccion que se usara para training: \n');
[train_indices, test_indices] = get_percentage(frac, Trainnumbers);
train_data.images = Trainnumbers.image(:, train_indices);
train_data.labels = Trainnumbers.label(train_indices);
test_data.images = Trainnumbers.image(:, test_indices);
test_data.labels = Trainnumbers.label(test_indices);

% Convertir los datos de entrenamiento y prueba al formato adecuado
train_data.images = reshape(train_data.images, [28, 28, 1, numel(train_indices)]);
test_data.images = reshape(test_data.images, [28, 28, 1, numel(test_indices)]);

% Definir la arquitectura de la red neuronal
layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    % Agrega una capa de convolucion de 64
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

% Opciones de entrenamiento
options = trainingOptions(['adam' ...
    ''], ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 20, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Entrenar la red neuronal
net = trainNetwork(train_data.images, categorical(train_data.labels), layers, options);

% Clasificar los datos de prueba
predicted_labels = classify(net, test_data.images);

% Calcular la precisión
accuracy = sum(predicted_labels == categorical(test_data.labels')) / numel(test_data.labels);
fprintf('Precisión: %.2f%%\n', accuracy * 100);


