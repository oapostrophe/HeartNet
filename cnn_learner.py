import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *

torch.cuda.set_device(3)

image_path = Path('/raid/heartnet/data/test_cnn_learner')
images = get_image_files(image_path)

invalid_images = verify_images(images)
print(invalid_images)

recall_function = Recall(pos_label=0)
precision_function = Precision(pos_label=0)

images_datablock = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    batch_tfms=aug_transforms())

dls = images_datablock.dataloaders(image_path, bs=4)

learn = cnn_learner(dls, resnet18, metrics=[error_rate, recall_function, precision_function])

learn.fine_tune(3)
