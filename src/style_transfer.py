from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import time

import torch
import torch.nn
import torch.optim as optim
from torchvision import transforms, models

import models
import utils
from glob import glob
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('------------------------------------------------------------------')
print(device)
print('------------------------------------------------------------------')
# get the VGG19's structure except the full-connect layers
VGG = models.vgg19(pretrained=True).features
VGG.to(device)
print(VGG)
# only use VGG19 to extract features, we don't need to change it's parameters
for parameter in VGG.parameters():
    parameter.requires_grad_(False)

    


contents = glob("../input/contents/*")
styles = glob("../input/styles/*")
style_loss_weights = [1,15,50]

counter = 0
for content in contents:
    cname = content.split("/")[-1][:-4]
    os.makedirs("output/{}".format(cname),exist_ok=True)
    
    for style in styles:
        

        
        sname = style.split("/")[-1][:-4] 
        
        for style_weight in style_loss_weights:
   
            print(sname,cname)
            style_net = models.TransformationNet()
            style_net.to(device)
            content_image = utils.load_image(content, img_size=500)  # temporary/content.png
            content_image = content_image.to(device)


            style_image = utils.load_image(style, img_size=500)  # temporary/style.png
            style_image = style_image.to(device)
            style_image.shape,content_image.shape

            content_features = utils.get_features(content_image, VGG)
            style_features   = utils.get_features(style_image, VGG)

            style_gram_matrixs = {layer: utils.get_grim_matrix(style_features[layer]) for layer in style_features}

            target = content_image.clone().requires_grad_(True).to(device)

            # try to give fore con_layers more weight so that can get more detail in output iamge
            style_weights = {'conv1_1': 0.1,
                            'conv2_1': 0.2,
                            'conv3_1': 0.4,
                            'conv4_1': 0.8,
                            'conv5_1': 1.6}

            content_weight = 150
            # style_weight = 

            # show_every = 10000
            optimizer = optim.Adam(style_net.parameters(), lr=5e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
            steps = 1000

            content_loss_epoch = []
            style_loss_epoch = []
            total_loss_epoch = []
            output_image = content_image

            time_start=time.time()
            for epoch in range(0, steps+1):
                
                scheduler.step()

                target = style_net(content_image).to(device)
                target.requires_grad_(True)


                target_features = utils.get_features(target, VGG)  # extract output image's all feature maps
                content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
                
                style_loss = 0

                # compute each layer's style loss and add them
                for layer in style_weights:
                    
                    target_feature = target_features[layer]  # output image's feature map after layer
                    target_gram_matrix = utils.get_grim_matrix(target_feature)
                    style_gram_matrix = style_gram_matrixs[layer]

                    layer_style_loss = style_weights[layer] * torch.mean((target_gram_matrix - style_gram_matrix) ** 2)
                    b, c, h, w = target_feature.shape
                    style_loss += layer_style_loss / (c * h * w)
                

                total_loss = content_weight * content_loss + style_weight * style_loss
                total_loss_epoch.append(total_loss)

                style_loss_epoch.append(style_weight * style_loss)
                content_loss_epoch.append(content_weight * content_loss)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

 

                output_image = target
            time_end=time.time()
            print('totally cost', time_end - time_start)

            # display the raw images
            fig, (ax1) = plt.subplots(1, figsize=(20, 10))
            # content and style ims side-by-side
            # ax1.imshow(utils.im_convert(content_image))
            ax1.imshow(utils.im_convert(output_image))
            plt.savefig("output/{}/{}_{}.jpg".format(cname,sname,style_weight))
            plt.show()

            
