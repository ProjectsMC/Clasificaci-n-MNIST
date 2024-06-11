clc
clear all
load ('Trainnumbers.mat'); 
x = input( 'Ingrese la dimension a la que busca reducir' ) %Dimension a reducir

% Aca se divide de entrada los datos de training y test
frac= input( 'Ingrese la fraccion que se usara para training: \n' );
[train_indices,test_indices]=get_percentage_A(frac,Trainnumbers);
train_data.images=(Trainnumbers.image(:,train_indices));
train_data.labels=(Trainnumbers.label(train_indices));
test_data.images=(Trainnumbers.image(:,test_indices));
test_data.labels=(Trainnumbers.label(test_indices));

% Aca normalizamos las imagenes y se adjuntan a la base de datos de cada
% set
[train_data.images_n,test_data.images_n]=normalize(train_data,test_data);

[train_data.im_PCA,test_data.im_PCA,outmse]=PCA_F(x,train_data.images,test_data.images);


tami = length(train_data.im_PCA);
classes=train_data.labels;
uClass= unique(classes);
classlen=size(uClass);

% Definir parámetros del SOM
dimension=[16,16];
gridSize = [16, 16]; % Tamaño de la red (10x10 neuronas)
topologyFcn = 'hextop'; % Topología hexagonal
distanceFcn = 'linkdist'; % Función de distancia

% Crear el SOM con los parámetros definidos
somnet = selforgmap(gridSize, 100, 3, topologyFcn, distanceFcn);

% Configurar parámetros de entrenamiento
somnet.trainParam.epochs = 200; % Número de iteraciones
somnet.trainParam.lr = 0.1; % Tasa de aprendizaje inicial


% %Entrenando la red
% [~,n ]=size(train_data.im_PCA);
% indx=randperm(n);

netsom = train(somnet,train_data.im_PCA);

%Imprmir distribucion de red de forma diferente
% figure,hold on
% plotsomnd(netsom)  
% ynt=netsom(t.value);
ynt_t=netsom(train_data.im_PCA);


neurons=size(ynt_t);  %Vector con el número de neuronas
vector_class=zeros(length(uClass),neurons(1));
yntind=vec2ind(ynt_t);
ynt_label=zeros(2,tami(1));  %Matriz relación clase e indice de neurona ganadora
for i= 1:tami(1)
    ynt_label(:,i)=[classes(i);yntind(i)];
end

%Obteniendo número de neuronas ganadoras
for i=1:tami(1)
     if ynt_label(1,i)==uClass(1)
            vector_class(1,ynt_label(2,i))=vector_class(1,ynt_label(2,i))+1;
     elseif ynt_label(1,i)==uClass(2)
             vector_class(2,ynt_label(2,i))=vector_class(2,ynt_label(2,i))+1;
     elseif ynt_label(1,i)==uClass(3)
             vector_class(3,ynt_label(2,i))=vector_class(3,ynt_label(2,i))+1;
     elseif ynt_label(1,i)==uClass(4)
             vector_class(4,ynt_label(2,i))=vector_class(4,ynt_label(2,i))+1;
     elseif ynt_label(1,i)==uClass(5)
             vector_class(5,ynt_label(2,i))=vector_class(5,ynt_label(2,i))+1;
     elseif ynt_label(1,i)==uClass(6)
             vector_class(6,ynt_label(2,i))=vector_class(6,ynt_label(2,i))+1;
     elseif ynt_label(1,i)==uClass(7)
             vector_class(7,ynt_label(2,i))=vector_class(7,ynt_label(2,i))+1;
     elseif ynt_label(1,i)==uClass(8)
             vector_class(8,ynt_label(2,i))=vector_class(8,ynt_label(2,i))+1;
     elseif ynt_label(1,i)==uClass(9)
             vector_class(9,ynt_label(2,i))=vector_class(9,ynt_label(2,i))+1;
     else 
             vector_class(10,ynt_label(2,i))=vector_class(10,ynt_label(2,i))+1;
     end
end

% 
% %Etiquetando las neuronas
neurons_label=zeros(1,neurons(1));
for j=1:neurons(1)
        [mayorValor, indiceMayorValor] = max(vector_class(:,j)');
        neurons_label(j)=uClass(indiceMayorValor);

end

% %Obteniendo los indices ganadores
% yntind_test=vec2ind(ynt_t);
% %Realizando la predicción
% y_pred=zeros(1,tami(1));
% for k=1:tami(1)
%     y_pred(k)=neurons_label(yntind_test(k));
% end

% %Obteniendo la prediccion

y_test=netsom(test_data.im_PCA);
y_test_ind=vec2ind(y_test);

% Inicializar y_pred
y_pred = [];

% Iterar sobre cada elemento en y_test_ind
for i = 1:length(y_test_ind)
    block = y_test_ind(i);
    % Iterar sobre cada elemento en neurons_label
    for j = 1:length(neurons_label)
        neuron = neurons_label(j);
        if block == j
            % Añadir el valor de neuron a y_pred
            y_pred = [y_pred, neuron];
        end
    end
end

% Mostrar el resultado
disp('y_pred:');
disp(y_pred);
% %Valores mal clasificado
misclassifications_t = sum(y_pred ~= test_data.labels);
disp(misclassifications_t);
% 
% %Sacando matriz de confusión
C=confusionmat(test_data.labels,y_pred)
figure
cm=confusionchart(test_data.labels,y_pred)

[accuracy_SOM,misclassifications_test_SOM]=evaluation(y_pred, test_data.labels);
fprintf('\nLa precisión usando SOM  %.2f%%\n', accuracy_SOM * 100);
fprintf('Misclassifications usando SOM   %.2f%%\n', misclassifications_test_SOM);

figure   
plotsomnd(netsom,train_data.im_PCA)


% Step 3: Plot the SOM with colored labels
figure;
hold on;
colors = lines(max(neurons_label(:))); % Generate a set of colors for the labels

% Get the positions of the neurons
positions = netsom.layers{1}.positions';

% Scatter plot for each label with different colors
for i = 1:dimension(1)
    for j = 1:dimension(2)
        label = labels(i, j); % Get the label of the neuron
        color = colors(label, :); % Get the color associated with the label
        scatter(positions(1, (i-1)*dimension(2) + j), positions(2, (i-1)*dimension(2) + j), 100, color, 'filled');
    end
end
