from PIL import Image
import numpy as np
im = Image.open('data/img/doge.jpg')
im.show()
im = np.array(im)
print(im.shape)