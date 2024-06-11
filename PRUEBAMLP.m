clear all
clc
load ('Trainnumbers.mat'); 
x = input( 'Ingrese la dimension a la que busca reducir' ) %Dimension a reducir

% Aca se divide de entrada los datos de training y test
frac= input( 'Ingrese la fraccion que se usara para training: \n' );
[train_indices,test_indices]=get_percentage(frac,Trainnumbers);
train_data.images=(Trainnumbers.image(:,train_indices));
train_data.labels=(Trainnumbers.label(train_indices));
test_data.images=(Trainnumbers.image(:,test_indices));
test_data.labels=(Trainnumbers.label(test_indices));

% Aca normalizamos las imagenes y se adjuntan a la base de datos de cada
% set
[train_data.images_n,test_data.images_n]=normalize(train_data,test_data);

[train_data.im_PCA,test_data.im_PCA,outmse,W]=PCA_F(x,train_data.images,test_data.images);

%Codificanco labels
% Número de clases
numClasses = 10;

% Inicializar la matriz one-hot con ceros
labelscodtrain = zeros(length(train_data.labels), numClasses);

% Asignar uno en la posición correspondiente a cada etiqueta
for i = 1:length(train_data.labels)
    idx=train_data.labels(i);
    if idx==0
        idx=10;
    end
    labelscodtrain(i, idx) = 1;
end

% Mostrar la matriz one-hot
disp(labelscodtrain);

% Inicializar la matriz one-hot con ceros
labelscodtest = zeros(length(test_data.labels), numClasses);

% Asignar uno en la posición correspondiente a cada etiqueta
for i = 1:length(test_data.labels)
    idy=test_data.labels(i);
    if idy==0
        idy=10;
    end
    labelscodtest(i, idy) = 1;
end

% Mostrar la matriz one-hot
disp(labelscodtest);

vector=[40:10:120]
accuracyglob=[]
for hs=40:10:120

    
    % Definir la estructura de la red MLP
    hiddenSizes = [148];  % Número de neuronas en la capa oculta
    activation = 'relu'; % Función de activación para la capa oculta
    solver = 'Adam'; % Algoritmo de optimización
    max_iter = 200; % Número máximo de iteraciones para el entrenamiento
    
    % Crear y entrenar la red MLP con capa softmax en la salida
    net = patternnet(hiddenSizes, 'trainscg');
    net.layers{2}.transferFcn = 'softmax'; % Capa softmax en la salida
    
    % Set the training function to trainscg
    net.trainFcn = 'trainscg';  % Cambia 'trainlm' por 'trainscg'
    net.trainParam.epochs = max_iter;  % Número de iteraciones
    % % Configurar la división de los datos
    % net.divideParam.trainRatio = 0.7;
    % net.divideParam.valRatio = 0.15;
    % net.divideParam.testRatio = 0.15;
    
    % Entrenar la red
    [net, tr] = train(net, train_data.im_PCA, labelscodtrain');
    
    % Evaluar la red MLP
    y = net(test_data.im_PCA); 
    yi = net(train_data.im_PCA); 
    
    
    % Inicializar vectores para las etiquetas predichas y verdaderas
    num_samples = size(yi, 2);
    num_test = size(y, 2);
    pred = zeros(1,num_samples);
    gtlabel = zeros(1,num_test);
    
    % Encontrar el índice con el valor máximo para cada muestra
    for i = 1:num_samples
    [~,pred(i)] = max(yi(:,i));
    if pred(i)==10
        pred(i)=0;
    end
    end
    for j = 1:num_test
    [~,gtlabel(j)] = max(y(:,j));
     if gtlabel(j)==10
        gtlabel(j)=0;
    end
    end
    
    % Visualizar los resultados
    disp('Predicted Labels:');
    disp(pred');
    disp('Ground Truth Labels:');
    disp(gtlabel');
    
    % Calcular errores de entrenamiento y test
    trainerror = perform(net, train_data.labels,pred);
    testerror = perform(net, test_data.labels,gtlabel);
    
    % Mostrar errores de entrenamiento y test
    fprintf('Error de entrenamiento: %.4f\n', trainerror);
    fprintf('Error de test: %.4f\n', testerror);
    
    % Mostrar precisión
    train_accuracy = 1 - trainerror;
    test_accuracy = 1 - testerror;
    
    [accuracy,miss_classification]=evaluation(gtlabel,test_data.labels);
    accuracyglob=[accuracyglob,accuracy];
    fprintf('\nLa precisión test del MLP: %.2f%%\n', accuracy* 100);
    
    fprintf('Precisión de entrenamiento: %.4f\n', train_accuracy);
    fprintf('Precisión de test: %.4f\n', test_accuracy);
end

% Graficar errores
figure;
plot(vector,accuracyglob);
xlabel('Hidden layer size');
ylabel('Accuracy');
title('Neuronas capa oculta vs accuracy')
grid on;



