import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *

# Pick a GPU with free resources (change this accordingly)
torch.cuda.set_device(0)

# Get images
image_path = Path('/raid/heartnet/data/imgset2')
images = get_image_files(image_path)

# Initialize metric functions
recall_function = Recall(pos_label=0)
precision_function = Precision(pos_label=0)
f1_score = F1Score(pos_label=0)

# Initialize DataLoader
images_datablock = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    batch_tfms=aug_transforms(do_flip=False)
)
dls = images_datablock.dataloaders(image_path, bs=16)

# Create, train, and save model
learn = cnn_learner(dls, resnet152, metrics=[error_rate, recall_function, precision_function, f1_score])
learn.fine_tune(16)
learn.export('demo_model_50.pkl')
