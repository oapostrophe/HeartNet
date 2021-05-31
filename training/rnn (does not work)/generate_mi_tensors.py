import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
import shutil


mi_src = "./imgset_rnn/mi/"
mi_dest = "./tensorfiles_rnn/mi"

tensornum = 0

for dirs, sub_dirs, files in os.walk(mi_src):
    if (dirs != "./imgset_rnn/mi/"):
        tensors = []
        pathlist = Path(dirs).glob('**/*.png')
        for path in pathlist:
            path_in_str = str(path) # because path is object not string
            #print(path_in_str)
            
            pil_img = Image.open(path_in_str).convert("RGB")
            pil_to_tensor = transforms.ToTensor()(pil_img)
            tensors.append(pil_to_tensor)

        tensor = torch.stack(tuple(tensors))
        tensor2 = torch.unsqueeze(tensor, 0)
        
        tensorfile = torch.save(tensor2, 'tensor' + str(tensornum) + '.pt')
  
        shutil.move('tensor' + str(tensornum) + '.pt', './tensorfiles_rnn/mi/tensor' + str(tensornum) + '.pt') 
        print('moved ' + str(tensornum) + '!')
        tensornum += 1


#generated 5486 mi tensors :) 
