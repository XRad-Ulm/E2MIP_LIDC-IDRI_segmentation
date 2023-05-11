import os
import shutil
import torch
from torchmetrics import Dice
import numpy as np
from model import UNet3D
import nibabel as nib


def test_model(dataloader, args):
    model = UNet3D()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    if os.path.isdir("testing_data_prediction_segmentation"):
        shutil.rmtree('testing_data_prediction_segmentation')
    os.makedirs("testing_data_prediction_segmentation")
    test_fnc_final(model, dataloader, args)


def test_fnc_final(test_model, data_loader, args):
    test_model.eval()
    with torch.no_grad():
        for i, (x, scan_ID, orig_size) in enumerate(data_loader):
            if not os.path.isdir("testing_data_prediction_segmentation/scan_" + str(scan_ID[0][0].item())):
                os.makedirs("testing_data_prediction_segmentation/scan_" + str(scan_ID[0][0].item()))

            x = x.to("cuda", dtype=torch.float)
            x[x < -1000] = -1000
            x[x > 1000] = 1000
            x = (x - (-1000)) / (1000 - (-1000))

            preds = []
            for pi in np.arange(start=0, stop=x.shape[0], step=args.batch_size):
                if (pi + args.batch_size) >= x.shape[0]:
                    pred_pi_seg = test_model(x[pi:])
                else:
                    pred_pi_seg = test_model(x[pi:pi + args.batch_size])
                preds.append(pred_pi_seg)
            pred_seg = torch.asarray(preds[0])
            for allpreds in range(len(preds) - 1):
                pred_seg = torch.cat([pred_seg, torch.asarray(preds[allpreds + 1])], dim=0)
            depth_step = int(pred_seg.shape[0] / args.patch_size[0])
            plot_vol_seg = np.zeros((512, 512, depth_step * args.patch_size[2]))
            counteridx = 0
            for yi in range(8):
                for xi in range(8):
                    for depthi in range(depth_step):
                        plot_vol_seg[yi * args.patch_size[0]:(yi + 1) * args.patch_size[0], xi * args.patch_size[1]:(xi + 1) * args.patch_size[1],
                        depthi * args.patch_size[2]:(depthi + 1) * args.patch_size[2]] = pred_seg[counteridx, 0, :, :, :].cpu()
                        counteridx += 1
            plot_vol_seg = plot_vol_seg[:, :, :orig_size[0][-1]]
            pred_seg_nii = nib.Nifti1Image(plot_vol_seg, affine=np.eye(4))
            nib.save(pred_seg_nii,
                     "testing_data_prediction_segmentation/scan_" + str(scan_ID[0][0].item()) + "/prediction_total.nii")
            print("saved prediction " + str(i) + "/" + str(len(data_loader)))
        del x
        del preds
        del pred_seg
        del plot_vol_seg
        del pred_seg_nii

def calculateDice(args):
    dc = 0.0
    counter = 0
    dice = Dice(average='micro',ignore_index=0)
    for scan_file in os.listdir(args.testing_data_solution_path):
        y_mask = torch.from_numpy(
            nib.load(args.testing_data_solution_path + "/" + scan_file + "/segmentation_total.nii").get_fdata()).int()
        pred_seg = torch.from_numpy(
            nib.load("testing_data_prediction_segmentation/" + scan_file + "/prediction_total.nii").get_fdata())
        print("Dice score of " + str(scan_file) + ": " + str(dice(pred_seg, y_mask).item()))
        dc += dice(pred_seg, y_mask)
        counter += 1

    return dc / counter
