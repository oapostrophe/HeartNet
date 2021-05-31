# Adds shadows to the dataset

import cv2
import albumentations as A
from matplotlib import pyplot as plt
import automold
from random import randint

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)


shadow_dimension = 5

for pt_number in range(1, 21838):
    image = cv2.imread("/raid/heartnet/data/imgset3/normal/pt_"+str(pt_number)+".png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    roi = (0, 0, image.shape[1], image.shape[0])
    num_shadows = randint(1, 3)
    shadowy_image = automold.add_shadow(image, num_shadows, roi, shadow_dimension)
    shadowy_image = cv2.cvtColor(shadowy_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("/raid/heartnet/data/imgset3/shadow/pt_"+ str(pt_number)+".png", shadowy_image)
