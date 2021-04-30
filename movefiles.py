from PIL import Image
from pathlib import Path

# compress normal images
# https://stackoverflow.com/questions/10607468/how-to-reduce-the-image-file-size-using-pil

directory_in_str = "../imgset2/mi"

pathlist = Path(directory_in_str).glob('**/*.png')
counter = 0

for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    print(path_in_str, path.name)

    foo = Image.open(path_in_str)
    # reduce the size of the image,
    # I downsize the image with an ANTIALIAS filter (gives the highest quality)
    # foo = foo.resize((400,400),Image.ANTIALIAS)

    foo.save("./imgset_temp/mi_temp/"+path.name)
    # foo.save("./normalvmi/reducednorm/"+path.name,optimize=True,quality=95)
    counter += 1
    if counter == 500:
      break
    # The saved downsized image size is 22.9kb

# # compress mi images
# directory_in_str_mi = "./normalvmi/mi"

# pathlist_mi = Path(directory_in_str_mi).glob('**/*.png')

# for path in pathlist_mi:
#      # because path is object not string
#      path_in_str = str(path)
#      # print(path_in_str)

#     foo = Image.open(path_in_str)
#     # reduce the size of the image,
#     # I downsize the image with an ANTIALIAS filter (gives the highest quality)
#     foo = foo.resize((400,400),Image.ANTIALIAS)

#     foo.save("./normalvmi/reducedmi",optimize=True,quality=95)
#     # The saved downsized image size is 22.9kb