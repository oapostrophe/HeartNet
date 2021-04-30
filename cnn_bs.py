#import fastbook
#fastbook.setup_book()
#from fastbook import *
#from fastai import *
#from fastai.vision.widgets import *
from fastai.vision.all import *
import torch
import torchvision

torch.cuda.set_device(2)

image_path = Path('./imgset_temp')
images = get_image_files(image_path)

invalid_images = verify_images(images)
print(invalid_images)

images_datablock = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    batch_tfms=aug_transforms())

dls = images_datablock.dataloaders(image_path, bs=14)
#dls = images_datablock.dataloaders(image_path, bs=12, items_tfms=Resize(400))
learn = cnn_learner(dls, resnet34, metrics=error_rate)

learn.fine_tune(20)