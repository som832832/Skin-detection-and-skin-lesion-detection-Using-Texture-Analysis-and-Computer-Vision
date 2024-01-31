clc
clear
close all
format shortG

%% set Parameters

Algorithm = 'PSO';
ImagesPath = 'Images';
AUC_Calculate = 1;
Mode = 'N'; % 'L' for Linear Color Space - 'N' for nonLinear Color Space
load(['Data/Wmatrix_', Mode, '_', Algorithm, '.mat']);

nImages = 1; % number of Images
PlotRange = [1,1]; % Range for plot

%% Create Data

[Skin_RGB_Data,nonSkin_RGB_Data] = CreateData();

%% New Color Space

Skin_Data = rgb2newColorSpace(double(Skin_RGB_Data)'/255,W,Mode);
Skin_Data = Skin_Data';

nonSkin_Data = rgb2newColorSpace(double(nonSkin_RGB_Data)'/255,W,Mode);
nonSkin_Data = nonSkin_Data';

ColorSpaceFactor = 1;

%% Create input and output vector

TrainInputs = [Skin_Data,...
               nonSkin_Data]'; 
TrainInputs = double(TrainInputs)/ColorSpaceFactor; % normalize input vector between [0 1]

TrainTargets = [ones(1,size(Skin_Data,2)),...
                zeros(1,size(nonSkin_Data,2))]';

TrainData = [TrainInputs,TrainTargets];
TrainData = TrainData(randperm(length(TrainData)),:);

%% Design ANFIS

Exponent=2;
MaxIter=100;
Maximprovement=1e-2;
DisplayValue=0;
FCMOption=[Exponent ...
           MaxIter ...
           Maximprovement ...
           DisplayValue];
nRules=15;        
fis=genfis3(TrainInputs,TrainTargets,'sugeno',nRules,FCMOption);

MaxEpoch=1000;
ErrorGoal=0;
InitialStepSize=0.05;
StepSizeDecreaseRate=0.9;
StepSizeIncreaseRate=1.1;
TrainOptions=[MaxEpoch ...
              ErrorGoal ...
              InitialStepSize ...
              StepSizeDecreaseRate ...
              StepSizeIncreaseRate];

DisplayInfo=true;
DisplayError=true;
DisplayStepSize=true;
DisplayFinalResult=true;
DisplayOptions=[DisplayInfo ...
                DisplayError ...
                DisplayStepSize ...
                DisplayFinalResult];

OptimizationMethod=1;
% 0: Backpropagation
% 1: Hybrid
    
%% Train ANFIS

fis=anfis(TrainData,fis,TrainOptions,DisplayOptions,[],OptimizationMethod);

%% Apply ANFIS to Train Data

TrainOutputs=evalfis(TrainInputs,fis);

figure;
PlotResults(TrainTargets,TrainOutputs,'Train Data');

%% Test Data

% Input Images
Images = cell(nImages,1);
for i=1:nImages
   Images{i} = imread([ImagesPath, '\', 'P', num2str(i), '.jpg']); 
end

% Mask Images
Masks = cell(nImages,1);
for i=1:nImages
   Mask = imread([ImagesPath, '\', 'P', num2str(i), '_M', '.jpg']);
   
   if(size(Mask,3)==3)
      Mask = rgb2gray(Mask);
   end
   
   Masks{i} = im2bw(Mask,0.5);
end

% TestInputs
TestInputs = cell(nImages,1);
for i=1:nImages     
    [row,col,~] = size(Images{i});
    imR = Images{i}(:,:,1);
    imG = Images{i}(:,:,2);
    imB = Images{i}(:,:,3);
    imR = reshape(imR,1,[]);
    imG = reshape(imG,1,[]);
    imB = reshape(imB,1,[]);
    im_3xn = [imR;imG;imB];
    
    Im_Data_3xn = rgb2newColorSpace(double(im_3xn)'/255,W,Mode);
    Im_Data_3xn = Im_Data_3xn';
    
    TestInputs{i} = double(Im_Data_3xn)/ColorSpaceFactor; % normalize between [0 1]
end

%% Test ANFIS

TestImage = cell(nImages,1);
Scores = [];
Resp = [];
EvalTime = 0;

for i=1:nImages
   tic; 
   Score = evalfis(TestInputs{i},fis);
   t = toc;
   EvalTime = EvalTime+t;
   
   Score = max(min(Score,1),0);
   
   TestOutputs=(Score>0.5);
   TestImage{i}=reshape(TestOutputs,[size(Images{i},1),size(Images{i},2)]);
   
   if(AUC_Calculate==1)            
       Scores = [Scores;Score]; %#ok
       M = Masks{i};
       Resp = [Resp;M(:)]; %#ok
   end
end

EvalTime = EvalTime/nImages;

%% Calculate ROC

[X,Y,T,AUC,OPTROCPT] = perfcurve(Resp,Scores,1);

for i=1:length(X)
    Point = X(i)+Y(i);
    if(Point>=1)
       EER_1 = Y(i); 
       break; 
    end
end

figure;
plot(X,Y);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC for Classification by ANFISn');

%% Show Outputs

se=strel('disk',4);

for i=(PlotRange(1):PlotRange(2))
   figure;
   Original Mask
   subplot(2,3,1),imshow(Masks{i});
   title('Original Mask');
   
    Original Imag
  subplot(2,3,2),imshow(Images{i});
   title('Original Image');
 
   % New Color Space Image
   [r,c,~] = size(Images{i});
   RGB = reshape(Images{i},r*c,3);
   I_New = rgb2newColorSpace(double(RGB)/255,W,Mode);
   I_New = reshape(I_New,r,c,3);
   subplot(2,3,3),imshow(mat2gray(I_New));
   title('New Color Space Image');
   
   % ANFIS Mask
   subplot(2,3,4),imshow(TestImage{i});
   title('ANFIS Mask');
   
  % subplot(1,1,1),imshow(ANFIS Mask);
   %title('ANFIS Mask');  
   %imwrite(ANFIS Mask,'ANFIS Mask.jpg')

   % ANFIS Output
   imc=imclose(TestImage{i},se);
   imco=imopen(imc,se);
   imco=repmat(imco,1,1,3);
   OutImage_RGB=uint8(double(Images{i}).*imco);
   subplot(2,3,5),imshow(OutImage_RGB);
   title('ANFIS Output');   
   saveas(gcf,[ImagesPath, '\', 'P', num2str(i), '_Out', '.jpg']);
end

%% Calculate Evaluation

CDR = zeros(1,nImages);
FAR = zeros(1,nImages);
FRR = zeros(1,nImages);
R = zeros(1,nImages);
P = zeros(1,nImages);
F = zeros(1,nImages);
FPR = zeros(1,nImages);
FNR = zeros(1,nImages);
TNR = zeros(1,nImages);
TDE = zeros(1,nImages);
ACC = zeros(1,nImages);

for i=1:nImages
    A = Masks{i};
    B = TestImage{i};
    totalPixels = numel(A);

    Tr = sum(sum( (A==B) ));    
    TP = sum(sum( (A==B)&(B==1) ));
    TN = sum(sum( (A==B)&(B==0) ));    
    FP = sum(sum( (A~=B)&(B==1) ));
    FN = sum(sum( (A~=B)&(B==0) ));
    
    % Evaluation protocols
    CDR(i) = Tr/totalPixels; % correct detection rate 
    FAR(i) = FP/totalPixels; % false acceptance rate
    FRR(i) = FN/totalPixels; % false rejection rate
    
    R(i) = TP/(TP+FN);
    P(i) = TP/(TP+FP);
    F(i) = (2*P(i)*R(i))/(P(i)+R(i));    
    FPR(i) = FP/(TN+FP);
    FNR(i) = FN/(TP+FN);
    TNR(i) = TN/(TN+FP);
    TDE(i) = FPR(i)+FNR(i);
    ACC(i) = (TP+TN)/(TP+TN+FP+FN);
    IOU (i)=(TP/(TP+FN+FP));
end

clc;

disp('*************************************************');
disp(['Mean of CDR = ' num2str(mean(CDR))]);
disp(['Mean of FRR = ' num2str(mean(FRR))]);
disp(['Mean of FAR = ' num2str(mean(FAR))]);
disp(['Mean of R = ' num2str(mean(R))]);
disp(['Mean of P = ' num2str(mean(P))]);
disp(['Mean of F = ' num2str(mean(F))]);
disp(['Mean of FPR = ' num2str(mean(FPR))]);
disp(['Mean of FNR = ' num2str(mean(FNR))]);
disp(['Mean of TNR = ' num2str(mean(TNR))]);
disp(['Mean of TDE = ' num2str(mean(TDE))]);
disp(['Mean of ACC = ' num2str(mean(ACC))]);
disp(['Mean of AUC = ' num2str(AUC)]);
disp(['Mean of 1-EER = ' num2str(EER_1)]);
disp(['Mean of IOU = ' num2str(IOU)]);
disp(['Mean of Time (sec) = ' num2str(EvalTime)]);
disp('*************************************************');
