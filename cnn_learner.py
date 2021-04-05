import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *

torch.cuda.set_device(2)

image_path = Path('./data/imgset1/')
images = get_image_files(image_path)

invalid_images = verify_images(images)
print(invalid_images)

images_datablock = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    batch_tfms=aug_transforms())

dls = images_datablock.dataloaders(image_path, bs=16)

learn = cnn_learner(dls, resnet18, metrics=error_rate)

learn.fine_tune(4)