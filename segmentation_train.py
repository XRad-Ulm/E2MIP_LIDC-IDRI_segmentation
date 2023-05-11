import matplotlib.pyplot as plt
import torch
from torchmetrics import Dice
import datetime
from model import UNet3D
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def train_loop_seg(train_loader, val_loader, args):
    input_channels = next(iter(train_loader))[0].shape[1]
    print(input_channels)
    model = UNet3D()
    model.cuda()
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
    scheduler = ReduceLROnPlateau(optimizer, 'max')
    datestr = str(datetime.datetime.now())
    print("this run has datestr " + datestr)
    tr_dcs,tr_losses = [],[]
    val_dcs = []
    best_val = 0.0
    dice = Dice(average='micro',ignore_index=0).to("cuda")
    for ep in range(args.epochs):
        print("Epoch " + str(ep))
        print("Training")
        model, tr_dc, tr_loss = train_fnc(model, train_loader, optimizer, dice)
        print("tr_dc=" + str(tr_dc))
        tr_dcs.append(tr_dc.cpu().detach().numpy())
        tr_losses.append(tr_loss.cpu().detach().numpy())
        print("Validation")
        val_dc = val_fnc(model, val_loader,dice)
        scheduler.step(val_dc)
        print(get_lr(optimizer))
        val_dcs.append(val_dc.cpu().detach().numpy())
        print("val_dc=" + str(val_dc))
        if val_dc.cpu().detach().numpy() > best_val:
            torch.save({
                'epoch': ep + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, str(datestr) + ".model")
            print("Saving new best model ", str(datestr) + ".model")
            best_val = val_dc.cpu().detach().numpy()
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(args.epochs), tr_dcs)
    plt.title("Train Dice")
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(args.epochs), tr_losses)
    plt.title("Train Loss")
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(args.epochs), val_dcs)
    plt.title("Val Dice")
    plt.show()
    return str(datestr) + ".model"


def loss_fcn(gt, pred):
    L_seg = torch_dice_coef_loss(gt, pred)
    return L_seg


def torch_dice_coef_loss(y_true, y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))


def train_fnc(train_model, data_loader, optim, dicefnc):
    train_model.train()
    tr_dc = 0
    tr_loss = 0
    counter = 0
    for i, (x, y_mask) in enumerate(data_loader):
        x, y_mask = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float)
        if y_mask.max() > 0:

            x[x < -1000] = -1000
            x[x > 1000] = 1000
            x = (x - (-1000)) / (1000 - (-1000))

            pred_seg = train_model(x)

            loss = loss_fcn(y_mask, pred_seg)
            optim.zero_grad()
            loss.backward()
            optim.step()

            tr_loss += loss

            print("Training: " + str(round(i / len(data_loader), ndigits=3)) + " Dice: " + str(
                dicefnc(pred_seg, y_mask.int()).item()) + "    Loss: " + str(loss.item()))
            tr_dc += dicefnc(pred_seg, y_mask.int())
            counter += 1

            del loss
            del x
            del y_mask
            del pred_seg
        else:
            print("batch with no roi")
    return train_model, tr_dc / counter, tr_loss / counter


def val_fnc(val_model, data_loader,dicefnc):
    val_model.eval()
    dc = 0.0
    counter = 0
    with torch.no_grad():
        for i, (x, y_mask) in enumerate(data_loader):
            x, y_mask = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float)
            if y_mask.max()>0:
                x[x < -1000] = -1000
                x[x > 1000] = 1000
                x = (x - (-1000)) / (1000 - (-1000))

                pred_seg = val_model(x)

                print("Validation: " + str(round(i / len(data_loader), ndigits=3)) + " Dice: " + str(
                    dicefnc(pred_seg, y_mask.int()).item()))
                dc += dicefnc(pred_seg, y_mask.int())
                counter += 1

                del x
                del y_mask
                del pred_seg
    return dc / counter

