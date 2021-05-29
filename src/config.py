import torch

'''
set hyperparameter values
'''
path = r'C:\Users\Pooja\Documents\ML_projects\Muralscape\model'
im_size = 256
device = torch.device('cuda')
im_path = r'C:\Users\Pooja\Documents\ML_projects\Muralscape\assets\jimi.jpeg'
style_path = r'C:\Users\Pooja\Documents\ML_projects\Muralscape\assets\psy.jpg'
out_path = r'C:\Users\Pooja\Documents\ML_projects\Muralscape\samples'
model_path = r'C:\Users\Pooja\Documents\ML_projects\Muralscape\model'
epochs = 4000
a = 1
b = 0.04
lr = 0.002
