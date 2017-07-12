from PIL import Image
import numpy as np
import xlsxwriter
import pandas as pd

im_gs = Image.open("pan-card.jpg")
#greyscale
#im_gs = Image.open("pan-card.jpg").convert('L')
im_gs = im_gs.resize((128,128))
#If image is rotated
#im_gs = im_gs.rotate(90, expand=True)
im_gs.save('p-card.jpg')
