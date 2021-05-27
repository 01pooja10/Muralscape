import torch
import torch.nn as nn
from torch.nn.optim import Adam
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import config
import warnings
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings('ignore')

mod = models.vgg19(pretrained=True).features
print(mod)

class VGGModel(nn.Module):
    def __init__(self):
        super(VGGModel,self).__init__()
        self.model = models.vgg19(pretrained=True).features[:29]
        self.feat = ['0', '5', '10', '19', '28']

    def forward(self,x):
        features = []
        for l,ln in enumerate(self.model):
            x = l(x)
            if str(ln) in self.feat:
                features.append(x)
        return features

#image loading and preprocessing
def image_loader(path):
    img = Image.open(path)
    transform = transforms.Compose([transforms.Resize((config.im_size,config.im_size)),
                                    transforms.ToTensor()
                                    transforms.Normalize((0.5,),(0.5,))])
    img = transform(img).unsqueeze(0)
    img = img.to(config.device)
    return img

def g(img):
    gen = img.clone().requires_grad(True)
    return gen.to(config.device)

def style(path):
    img = Image.open(path).unsqueeze(0)
    pass

orig = image_loader(config.im_path)
g = g(orig)
style_im = style(config.style_path)

def train(vgg,opt,loss,scl):
    '''
    Function: Train the entire Neural Style Transfer model

    Parameters:
    vgg - instance of the VGG-19 model
    opt - Optimizer Adam
    loss - Loss initialized
    scl - Scaler to scale the loss

    Returns: Epochs, loss values and completes training
    '''
    vgg.eval()
    for epoch in range(config.epochs):
        original = vgg(orig)
        generated = vgg(g)
        style = vgg(style_im)
        sloss = 0
        oloss = 0
        for o,g,s in zip(original,generated,style):
            batch_size,channel,height,width = g.shape
            print(g.shape)
            oloss += torch.mean((g-o)**2)

            #gram/style matrix
    pass

def main():
    opt = Adam([g],lr=config.lr)
    model = VGGModel().to(config.device)
    loss = nn.LGBFS()
    scaler = GradScaler()
    pass

if __name__=='__main__':
    main()
