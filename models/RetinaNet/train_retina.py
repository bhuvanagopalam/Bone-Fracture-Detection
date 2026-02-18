
import numpy as np
import pandas as pd

import logging

import pandas as pd
logger = logging.getLogger(__name__)

import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from retinanet import model
from retinanet.dataloader import collater, Resizer, Augmenter, Normalizer, UnNormalizer
from tqdm import tqdm

from dataset import FracAtlasDataset
from evaluation_metric import calc_iou
import json




def train_model(model,optimizer, train_data_loader,lr_scheduler):

    model.train()
    
    epoch_loss = []
    running_loss =0

    for data in tqdm(train_data_loader):
                
        optimizer.zero_grad()

        classification_loss, regression_loss = model([data['img'].cuda().float(), data['annot'].cuda().float()])
                
        
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()

        loss = classification_loss + regression_loss
        
        loss.backward()

        optimizer.step()

        epoch_loss.append(float(loss))
        running_loss += loss.item()
        
        if lr_scheduler is not None:
            lr_scheduler.step()

    return running_loss/len(train_data_loader)

def validation_model(model, valid_data_loader):

    model.eval()
    
    epoch_loss = []
    IoU = 0
    num_pos = 0

    for data in tqdm(valid_data_loader):
                
        with torch.no_grad():
            if data['annot'][0,:, 4][0] == 0 or data['annot'][0, :, 4].shape[0] > 1 :
                continue
            scores, classification, transformed_anchors = model(data['img'].cuda().float())
            if len(scores) ==0:
                continue
            idxs= np.argmax(scores.cpu().detach().numpy())
            iou = calc_iou(transformed_anchors.cpu()[idxs].unsqueeze(0), data['annot'][0,:, :4])
            IoU += iou.item()
            num_pos += 1
            
            
            # classification_loss, regression_loss = model([data['img'].cuda().float(), data['annot'].cuda().float()])

            # classification_loss = classification_loss.mean()
            # regression_loss = regression_loss.mean()

            # loss = classification_loss +  regression_loss

            
            # epoch_loss.append(float(loss))
            
            # runnig_loss += loss.item()
        

    return IoU/ num_pos



def main():

    train_data = pd.read_csv("train_filtered.csv",)
    val_data= pd.read_csv("val_filtered.csv",)


    train_dataset = FracAtlasDataset(train_data, mode = "train", transforms = T.Compose([Augmenter(), Normalizer(), Resizer()]))
    valid_dataset = FracAtlasDataset(val_data, mode = "valid", transforms = T.Compose([Normalizer(), Resizer()]))


    train_data_loader = DataLoader(
        train_dataset,
        batch_size = 8,
        shuffle = True,
        num_workers = 16,
        collate_fn = collater,
        pin_memory=True
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 16,
        collate_fn = collater,
        pin_memory=True
    )


    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()

    retinanet = model.resnet50(num_classes = 2, pretrained = True).to(device)

    optimizer = torch.optim.AdamW(retinanet.parameters(), lr = 0.0001,weight_decay=1e-5)


    steps_per_epcoch = len(train_data_loader)
    epochs = 100
    tmax = epochs * steps_per_epcoch
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax,eta_min=1e-7,)

    train_loss = []
    val_iou = []
    best_iou = 0
    for epoch in tqdm(range(epochs)):
        
        tr_loss = train_model(retinanet, optimizer,train_data_loader, lr_scheduler)
        train_loss.append(tr_loss)
        
        
        if ((epoch + 1) % 10) == 0:
            iou = validation_model(retinanet, valid_data_loader)
            val_iou.append(iou)
            if iou> best_iou:
                best_iou = iou
                torch.save(retinanet, f"retinanet_pretrained_{best_iou:.3f}.pt")
            
            print(f'Val IoU:{iou}')


    with open('train_loss_pretrained.txt', 'w') as f:
        json.dump(train_loss, f)
    with open('val_iou_pretrained.txt', 'w') as f:
        json.dump(val_iou, f)
    
    print('Finished...')
    
if __name__ == "__main__":
    main()