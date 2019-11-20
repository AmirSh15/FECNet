### imports
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from models.FECNet import FECNet
from utils.pytorchtools import EarlyStopping
from utils.data_prep import DATALoader


### functions
def triplet_loss(y_true, y_pred):
    
    ref = y_pred[0::3,:]
    pos = y_pred[1::3,:]
    neg = y_pred[2::3,:]
    L12 = (ref-pos).pow(2).sum(1)
    L13 = (ref-neg).pow(2).sum(1)
    L23 = (pos-neg).pow(2).sum(1)
    correct = (L12<L13) * (L12<L23)

    alpha = 0.2
    d1 = F.relu((L12-L13)+alpha)
    d2 = F.relu((L12-L23)+alpha)
    d = torch.mean(d1+d2)
    return d, torch.sum(correct)





if __name__=='__main__':

    parser = argparse.ArgumentParser(description='PyTorch FECNet')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=240,
                        help='input batch size for training (default: 240)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of number of Validation data.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    model = FECNet()
    Num_Param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of Trainable Parameters= %d" % (Num_Param))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    early_stopping = EarlyStopping(patience=10, verbose=True)

    running_loss = 0
    print_per_epoch = 1
    correct = 0
    all = 0

    tr_dataloader, val_dataloader = DATALoader(csv_file=, val_ratio=args.val_ratio, num_workers=args.num_workers, batch_size=args.batch_size)

    # model.load_state_dict(torch.load('checkpoint_50_full_data.pt'))

    for epoch in range(args.epochs):
        # scheduler.step()

        # Training
        for i_batch, sample_batched in enumerate(tr_dataloader):
            model.zero_grad()
            # if(sample_batched is None) or (sample_batched.shape[0] == 0):
            #     print('Skip')
            #     continue
            targets = model(torch.FloatTensor(sample_batched).view(sample_batched.shape[0]*3, 3, 224, 224).cuda())

            loss, cor = triplet_loss(0, targets)
            all += sample_batched.shape[0]
            correct += cor.detach().cpu().numpy()

            loss.backward()
            optimizer.step()

            running_loss += loss.detach().cpu().numpy()

        if epoch % print_per_epoch == print_per_epoch - 1:  # print every 1 mini-batches

            # Validation
            all_val = 0
            correct_val = 0

            with torch.no_grad():
                running_loss_Valid = 0
                for i_batch, sample_batched in enumerate(val_dataloader):
 
                    # if (sample_batched is None) or (sample_batched.shape[0] == 0):
                    #     print('Skip')
                    #     continue
                    targets = model(torch.FloatTensor(sample_batched).view(sample_batched.shape[0]*3, 3, 224, 224).cuda())

                    loss, cor = triplet_loss(0, targets)
                    all_val += sample_batched.shape[0]
                    correct_val += cor.detach().cpu().numpy()
                    running_loss_Valid += loss.detach().cpu().numpy()

            print('[%d, %5d] loss: %.9f      Val_acc: %.5f    Train_acc: %.5f' % (epoch + 1, tr_dataset.__len__(),
                   running_loss / print_per_epoch, correct_val / all_val, correct/all))

            running_loss = 0
            all = 0
            correct = 0

            ### Check early stopping
            early_stopping(float(running_loss_Valid), model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        # if epoch%50 ==0:
            # torch.save(model.state_dict(), 'checkpoint_mine.pt')

    torch.save(model.state_dict(), 'checkpoint_200.pt')







