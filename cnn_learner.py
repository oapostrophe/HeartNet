import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *

torch.cuda.set_device(2)

image_path = Path('/raid/heartnet/data/imgset2')
images = get_image_files(image_path)

recall_function = Recall(pos_label=0)
# precision_function = Precision(pos_label=0)
# f1_score = F1Score(pos_label=0)

images_datablock = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    batch_tfms=aug_transforms(do_flip=False)
)
# do_flip=False in aug_transforms?
#item_tfms=Resize(400) under batch_tfms

dls = images_datablock.dataloaders(image_path, bs=16)

learn = cnn_learner(dls, resnet50, metrics=[error_rate, recall_function])

learn.fine_tune(16)

learn.export('demo_model_50.pkl')