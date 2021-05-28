import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from PIL import Image
import os
import config
from tqdm import tqdm
import warnings
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings('ignore')

mod = models.vgg19(pretrained=True).features

class VGGModel(nn.Module):
    def __init__(self):
        super(VGGModel,self).__init__()
        self.model = models.vgg19(pretrained=True).features[:29]
        self.feat = ['0', '5', '10', '19', '28']

    def forward(self,x):
        features = []
        for ln,l in enumerate(self.model):
            x = l(x)
            if str(ln) in self.feat:
                features.append(x)
        return features

#image loading and preprocessing
def image_loader(path):
    img = Image.open(path)
    transform = transforms.Compose([transforms.Resize((config.im_size,config.im_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))])
    img = transform(img).unsqueeze(0)
    img = img.to(config.device)
    return img

def gen(img):
    gen = img.clone().requires_grad_(True)
    return gen.to(config.device)

orig = image_loader(config.im_path)
gn = gen(orig)
style_im = image_loader(config.style_path)

def train(vgg,opt,scaler):
    '''
    Function: Train the entire Neural Style Transfer model

    Parameters:
    vgg - instance of the VGG-19 model
    opt - Optimizer Adam
    scaler - Scaler to scale loss for mixed precision training

    Returns: Epochs, loss values and completes training
    '''
    vgg.eval()
    for epoch in range(config.epochs):
        print('epoch ',str(epoch))
        original = vgg(orig)
        generated = vgg(gn)
        style = vgg(style_im)
        sloss = 0
        oloss = 0

        for o,g,s in zip(original,generated,style):
            batch_size,channel,height,width = g.shape
            oloss += torch.mean((g-o)**2)

            #gram/style matrix
            gram = g.view(channel,height*width).mm(g.view(channel,height*width).t())
            ab = s.view(channel,height*width).mm(s.view(channel,height*width).t())
            sloss = torch.mean((gram-ab)**2)


        with autocast():
            loss = config.a*oloss + config.b*sloss
            opt.zero_grad()
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(opt)
            scaler.update()

        if epoch%10==0:
            print('Model loss: ',loss.item())

    chkpt = {'model':vgg.state_dict(),'optimizer':opt.state_dict()}
    torch.save(chkpt,config.model_path+'/nst_model.pth')

    with torch.no_grad():
        save_image(gn,config.out_path+'/output1.jpg')

    print('Final Loss: ',loss.item())


def main():
    opt = optim.Adam([gn],lr=config.lr)
    model = VGGModel().to(config.device)
    scaler = GradScaler()
    train(model,opt,scaler)
    print('Training complete.')

if __name__=='__main__':
    main()
