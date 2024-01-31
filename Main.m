clc
clear
close all
format shortG

%% Set parameters

Algorithm = 'PSO';

ColorSpace = 'RGB'; % 'RGB' or 'Lab' or 'NewColorSpace'
Mode = 'L'; % 'L' for Linear Color Space - 'N' for nonLinear Color Space

load(['Data/Wmatrix_',Mode,'_',Algorithm,'.mat']);

ShowImages = 1;
LoadData = 1;

nImages = 15; % number of Images
MaskImage = 1; %Images that used as mask

addpath('Images');

%% Read Images

Images = cell(nImages,1);
for i=1:nImages
   Images{i} = imread([num2str(i) '.jpg']);
end

Im_mask = imread(['Images\' num2str(MaskImage) '_mask.jpg']);
Im_mask = im2bw(Im_mask); %#ok

[Skin_RGB_Data,nonSkin_RGB_Data] = CreateData();

%%  Select Color Space

switch ColorSpace  
   case 'RGB'
      Skin_Data = Skin_RGB_Data;
      nonSkin_Data = nonSkin_RGB_Data;
      
      ColorSpaceFactor = 255;
   case 'Lab'
      Skin_Data = rgb2lab(uint8(Skin_RGB_Data)');
      Skin_Data = Skin_Data';

      nonSkin_Data = rgb2lab(uint8(nonSkin_RGB_Data)');
      nonSkin_Data = nonSkin_Data';
      
      ColorSpaceFactor = 128;
   case 'NewColorSpace'
      Skin_Data = rgb2newColorSpace(double(Skin_RGB_Data)'/255,W,Mode);
      Skin_Data = Skin_Data';
      
      nonSkin_Data = rgb2newColorSpace(double(nonSkin_RGB_Data)'/255,W,Mode);
      nonSkin_Data = nonSkin_Data';
      
      ColorSpaceFactor = 1;
end

%% Create input and output vector

TrainInputs = [Skin_Data,...
               nonSkin_Data]'; 
TrainInputs = double(TrainInputs)/ColorSpaceFactor; % normalize input vector between [0 1]

TrainTargets = [ones(1,size(Skin_Data,2)),...
                zeros(1,size(nonSkin_Data,2))]';

TrainData = [TrainInputs,TrainTargets];
TrainData = TrainData(randperm(length(TrainData)),:);


%% configure Neural Network

if(LoadData==0)
   Exponent = 2;
   MaxIter = 100;
   Maximprovement = 1e-2;
   DisplayValue = 0;
   FCMOption  = [Exponent ...
                 MaxIter ...
                 Maximprovement ...
                 DisplayValue];
   nRules = 15;        
   fis = genfis3(TrainInputs,TrainTargets,'sugeno',nRules,FCMOption);

   MaxEpoch = 1000;
   ErrorGoal = 0;
   InitialStepSize = 0.05;
   StepSizeDecreaseRate = 0.9;
   StepSizeIncreaseRate = 1.1;
   TrainOptions  = [MaxEpoch ...
                    ErrorGoal ...
                    InitialStepSize ...
                    StepSizeDecreaseRate ...
                    StepSizeIncreaseRate];

   DisplayInfo = true;
   DisplayError = true;
   DisplayStepSize = true;
   DisplayFinalResult = true;
   DisplayOptions  = [DisplayInfo ...
                      DisplayError ...
                      DisplayStepSize ...
                      DisplayFinalResult];

   OptimizationMethod = 1; % 0: Backpropagation - 1: Hybrid
   fis = anfis(TrainData,fis,TrainOptions,DisplayOptions,[],OptimizationMethod);
   % fuzzy(fis);
   
   % Save
   if(strcmp(ColorSpace,'NewColorSpace'))
      % Save Network
      save(['Data/Anfis_',ColorSpace,'_',Mode,'_',Algorithm,'.mat'],'fis');
   else
      % Save Network
      save(['Data/Anfis_',ColorSpace,'.mat'],'fis');      
   end
else
   if(strcmp(ColorSpace,'NewColorSpace'))
      % Load Network
      load(['Data/Anfis_',ColorSpace,'_',Mode,'_',Algorithm,'.mat']);
   else
      % Load Network
      load(['Data/Anfis_',ColorSpace,'.mat']);      
   end
end

%% test Netwok by test set Images

Detect_Im_net = cell(1,nImages);

for i=1:nImages   
    if(ShowImages==true)
       figure;
       subplot(1,3,1),imshow(Images{i});
       title('original image');       
       switch ColorSpace
          case 'RGB'
             I_RGB = Images{i};
             subplot(1,3,2),imshow(I_RGB);
             title('RGB image');
          case 'Lab'
             I_Lab = rgb2lab(Images{i});
             subplot(1,3,2),imshow(mat2gray(I_Lab));
             title('Lab image');             
          case 'NewColorSpace'
             % convert color space
             [r,c,~] = size(Images{i});
             RGB = reshape(Images{i},r*c,3);
             I_New = rgb2newColorSpace(double(RGB)/255,W,Mode);
             I_New = reshape(I_New,r,c,3);
             subplot(1,3,2),imshow(mat2gray(I_New));
             title('New Color Space image');              
       end       
    end
    
    [row,col,~] = size(Images{i});
    imR = Images{i}(:,:,1);
    imG = Images{i}(:,:,2);
    imB = Images{i}(:,:,3);
    imR = reshape(imR,1,[]);
    imG = reshape(imG,1,[]);
    imB = reshape(imB,1,[]);
    im_3xn = [imR;imG;imB];
    
    % Select Color Space
    switch ColorSpace
       case 'RGB'
          Im_Data_3xn = im_3xn;   
          Im_Data_3xn = double(Im_Data_3xn)/ColorSpaceFactor; % normalize between [0 1]          
       case 'Lab'
          Im_Data_3xn = rgb2lab(im_3xn');
          Im_Data_3xn = Im_Data_3xn';          
          Im_Data_3xn = double(Im_Data_3xn)/ColorSpaceFactor; % normalize between [0 1]         
       case 'NewColorSpace'
          Im_Data_3xn = rgb2newColorSpace(double(im_3xn)'/255,W,Mode);
          Im_Data_3xn = Im_Data_3xn';          
          Im_Data_3xn = double(Im_Data_3xn)/ColorSpaceFactor; % normalize between [0 1] 
    end     

    % simulation ANFIS
    yout = evalfis(Im_Data_3xn',fis);
    yout(yout<=0.5) = 0;
    yout(yout>0.5) = 1;
    Detect_Im_net{i} = reshape(yout,row,col);
    if(ShowImages==true)
       subplot(1,3,3),imshow(Detect_Im_net{i});
       title('simulation Anfis');
    end
end

%% morphological processing and detected skin area

if(ShowImages==true)
   se=strel('disk',4);
   figure;

   for i=1:nImages
      imc = imclose(Detect_Im_net{i},se);
      imco = imopen(imc,se);
      imco = repmat(imco,[1,1,3]);
      im = uint8(double(Images{i}).*imco);
      subplot(4,floor( (size(Detect_Im_net,2)-1)/4 )+1,i);
      imshow(im);
   end
end

%% Calculate Accuracy

M = Im_mask;
AN = Detect_Im_net{MaskImage};

totalPixels = numel(Im_mask);

CDR = ( sum(sum( (M==AN) )) / totalPixels )*100; % correct detection rate 
FAR = ( sum(sum( (M~=AN)&(AN==1) )) / totalPixels )*100; % false acceptance rate
FRR = ( sum(sum( (M~=AN)&(AN==0) )) / totalPixels )*100; % false rejection rate

disp('*************************************************');
disp([ColorSpace,'_',Mode,' : ']);
disp(['CDR-net = ' num2str(mean(CDR))]);
disp(' ');
disp(['FRR-net = ' num2str(mean(FRR))]);
disp(' ');
disp(['FAR-net = ' num2str(mean(FAR))]);
disp('*************************************************');

%% Remove Paths

rmpath('Images');
