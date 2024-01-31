function Y=rgb2newColorSpace(RGBData,W,Mode)

   % convert color space
   % RGBData is : n*3
   % Y is : n*3
   
   switch Mode
      case 'L' % Linear
         Y = W*RGBData';
         Y = Y';
      case 'N'         
         Y = 0.5*(W*(RGBData*W)');
         Y = Y';
   end
   
end