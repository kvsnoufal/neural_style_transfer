# Neural Style Transfer using pyTorch
A CNN based approach to transfering "artistic style" from one image to another


Convolutional Neural Networks (CNN) are currently the state-of-the-art in computer vision applications, especially classification and object detection. In this article, we look at an unconventional application of CNNs - style transfer. This refers to the process of extracting the "artistic style" from an image S, and applying to a different image C, preserving the semantic contents of C.

![style_transfer](https://github.com/kvsnoufal/neural_style_transfer/blob/main/doc/1.png)

## Results

![results1](https://github.com/kvsnoufal/neural_style_transfer/blob/main/doc/2.jpeg)
![results2](https://github.com/kvsnoufal/neural_style_transfer/blob/main/doc/3.jpeg)
![results3](https://github.com/kvsnoufal/neural_style_transfer/blob/main/doc/4.jpeg)
![results5](https://github.com/kvsnoufal/neural_style_transfer/blob/main/doc/5.jpeg)

## How it's done?

CNNs are very powerful in classification applications because of their ability to extract features from images. The features extracted from image are usually lower dimensional representations of the image that encodes different attributes of an image - texture, color, semantic contents etc. Neural Style Transfer (NST) relies on carefully studying these features and successfully segregating the features that carry information about the style from the ones that contain information about the content.Once we are able to differentiate the 2, in theory, it is possible to combine the style from one image with the content of another image:

![img7](https://github.com/kvsnoufal/neural_style_transfer/blob/main/doc/7.png)


## Approach
The above style transfer can be achieved systematically by using CNNs and carefully designed feature extraction and loss functions.

![img8](https://github.com/kvsnoufal/neural_style_transfer/blob/main/doc/8.png)


---

#### Transformation Network
The transformation network is a generative neural network that takes an image as input and generates another image as output.
The weights of the transformation network are trained in a manner that:
 "content loss" between the input (content image) and the output image is minimized. This ensures that the semantic content in content image and output image is the same
"style loss" between style image and output image is minimized. This ensures that the style in style image is transferred to the output image

The output image would thus have the content from the content image and the style from style image.
The style and content loss are computed not on the images directly but on the features of the images extracted from a pre-trained neural network (VGG19 pre-trained on imagenet dataset). 
#### Content Loss
So, for calculating content loss we extract the features from a high enough layer of the pre-trained VGG model. Higher the layer, more abstract the features. The features contains information about perceptual content - like the outline of a building, main structures and outlines. The content loss is euclidean distance between the content features extracted from content and output images.
#### Style Loss
For calculating the style loss we extract the features from several layers of the VGG model. We then build a style representation for these features by calculating the Gram matrices (dot product of feature maps). This representation has shown to preserve the style of the image (texture, color etc.) without preserving the spatial structure/content. The style loss is the euclidean distance between the style representations of the style and output images.
### Training
The total loss to minimize in the training process is the weighted sum of the style and content losses. When the errors converge, the transformation network is capable of generating an image which maintains the spatial structure and perceptual content of the content image and the artistic style fo the style image.

### Shoulders of giants
1. A Neural Algorithm of Artistic Style - https://arxiv.org/pdf/1508.06576.pdf
2. Perceptual Losses for Real-Time Style Transfer and Super-Resolution - https://arxiv.org/pdf/1603.08155.pdf
3. AI epiphany youtube channel - https://www.youtube.com/watch?v=S78LQebx6jo&list=PLBoQnSflObcmbfshq9oNs41vODgXG-608
4. High-Resolution Network for Photorealistic Style Transfer
5. https://github.com/gordicaleksa/pytorch-neural-style-transfer
6. Code heavily "inspired" from: https://github.com/limingcv/Photorealistic-Style-Transfer

---
