
 I = imread('E:\FaceGallery\s1\1.pgm');
 imshow(I)
 %I = rgb2gray(RBG);
 
 FDetect = vision.CascadeObjectDetector;
 BB = step(FDetect,I);
 figure,imshow(I);
 rectangle('Position',BB,'LineWidth',4,'LineStyle','-','EdgeColor','y');
 title('Face Detection');
 face=imcrop(I,BB);
 figure,imshow(face);
 title('Cropped Face');
 
 normImage = mat2gray(face);
 figure,imshow(I);
 
%To detect Eyes
EyeDetect = vision.CascadeObjectDetector('EyePairBig');

%Read the input Image
I = imread('E:\FaceGallery\s1\1.pgm');

BB=step(EyeDetect,I);

figure,imshow(I);
rectangle('Position',BB,'LineWidth',4,'LineStyle','-','EdgeColor','b');
title('Eyes Detection');
Eyes=imcrop(I,BB);
figure,imshow(Eyes);
title('Eyes Cropped');
