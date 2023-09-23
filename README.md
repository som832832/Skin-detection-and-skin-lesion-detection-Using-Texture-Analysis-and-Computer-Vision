# Skin-detection-and-skin-lesion-detection-Using-Texture-Analysis-and-Computer-Vision
Skin detection and skin lesion detection Using Texture Analysis and Computer Vision

This paper deals with solving two problems in the diagnosis of pigmented skin lesions and human
skin detection. The first challenge is to diagnose pigmented skin lesions, including melanoma, in
complex images. Skin cancer incidence reached epidemic proportions and caused many deaths.
The issue with lesion image classification is that the segmentation does not correctly identify the
region of the lesion, and the features extracted are not very pertinent. The second problem is the
detection of human skin in low‑quality and obscene images. Skin color detection is an essential
step in various applications related to computer vision. These applications will include face detection,
finding ethnicity, age, diagnosis, and so on. In this paper, after pre‑processing using an
improved Gabor wavelet transform, the color and texture features of the images are extracted.
Using the PSO optimizer, the best texture and color features were selected for ANN and ANFIS
training. This paper has performed the detection of melanoma and other skin lesions as well as
the detection of skin in images in RGB, Lab, and an optimal and new color space created by FCM
and PSO. To solve the first problem, which is the diagnosis of skin lesions including melanoma,
two image datasets from the Atlas Dermoscopy and HAM 10000 databases have been used. To
solve the second problem, i.e., skin detection, three image datasets from the BAO, COMPAQ, and
Pratheepan databases have been used. Selecting the best color and texture features and testing
the ANFIS classifier in RGB, Lab, and the created optimal and new color space showed that this
method has good accuracy and speed for segmentation and classification of melanoma for skin cancer detection and also for skin detection in all three color spaces. The created optimal and
new color space had better performance than RGB and lab spaces for solving the above two challenges.
Skin detection’s results in the new proposed color space utilizing non‑linear conversion
by ANFIS classification have shown the highest accuracy with 89.22
optimal and new color space, which, compared to the best previous method on the Atlas Dermoscopy
database, accuracy has increased by 2This paper, through analysis and investigation,
showed that the selection and appropriate relationship between the best texture and color features
using the PSO optimizer is the primary key to increasing the accuracy of segmentation and
final classification. Choosing an appropriate color space and then choosing the classifier, respectively,
play the second and third roles in the final accuracy. Using PSO to select the best features
leads to improving the efficiency of the method as well as reducing its computational complexity.
Although the proposed method has higher accuracy and speed in creating an optimal and new
color space than RGB and LAB color spaces, according to the proper performance and accuracy
in these three color spaces, it can be concluded that the proposed method has low sensitivity
to the light intensity of the images. And it has practical applications in several datasets, such as
COMPAQ, Pratheepan, Atlas Dermoscopy, and HAM 10000.
