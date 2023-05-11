import numpy as np
import os
import torch
import nibabel as nib


def generateMyTrainingData(args):
    all_nodis_idx = 0
    scan_counter = 0
    for scan_folder in os.listdir(args.training_data_path):
        scan_counter += 1
        print("Creating \"my_training_data\" with custom preprocessed scan patches,  at scan: " + str(
            scan_counter) + " of " + str(len(os.listdir(args.training_data_path))))
        scan_vol = nib.load(args.training_data_path + "/" + scan_folder + "/image_total.nii").get_fdata()

        scan_segm = np.zeros_like(scan_vol)
        nodule_mean_centroids = np.empty((0,3))
        for nodule_folders in os.listdir(args.training_data_path + "/" + scan_folder):
            if os.path.isdir(args.training_data_path + "/" + scan_folder + "/" + nodule_folders):
                nodule_anni_centroids = np.empty((0,3))
                for nodule_annotation_folders in os.listdir(
                        args.training_data_path + "/" + scan_folder + "/" + nodule_folders):
                    nod_anni_mask = nib.load(
                        args.training_data_path + "/" + scan_folder + "/" + nodule_folders + "/" + nodule_annotation_folders + "/mask.nii").get_fdata().astype(
                        int)
                    nod_anni_centroid = np.loadtxt(
                        args.training_data_path + "/" + scan_folder + "/" + nodule_folders + "/" + nodule_annotation_folders + "/centroid.txt",
                        delimiter=',')
                    nod_anni_bbox = np.loadtxt(
                        args.training_data_path + "/" + scan_folder + "/" + nodule_folders + "/" + nodule_annotation_folders + "/bbox.txt",
                        delimiter=',')
                    nod_anni_bbox = (slice(int(nod_anni_bbox[0, 0]), int(nod_anni_bbox[0, 1])),
                                     slice(int(nod_anni_bbox[1, 0]), int(nod_anni_bbox[1, 1])),
                                     slice(int(nod_anni_bbox[2, 0]), int(nod_anni_bbox[2, 1])))
                    scan_segm[nod_anni_bbox] += nod_anni_mask
                    nodule_anni_centroids = np.vstack((nodule_anni_centroids,nod_anni_centroid))
                nodule_mean_centroid = np.mean(nodule_anni_centroids, axis=0).astype(int)
                nodule_mean_centroids = np.vstack((nodule_mean_centroids,nodule_mean_centroid))
        scan_segm[np.where(scan_segm > 0)] = 1
        if np.max(scan_segm) < 1:
            print("This scan has max segm val: " + str(np.max(scan_segm)))
            continue
        total_saved_patch_size = [128, 128, 128]
        for nodule_center in nodule_mean_centroids:
            cropout_border = np.array([[0, scan_vol.shape[0]], [0, scan_vol.shape[1]], [0, scan_vol.shape[2]]])

            for d in range(len(nodule_center)):
                if not (int(nodule_center[d] - (total_saved_patch_size[d] / 2)) < 0):
                    cropout_border[d, 0] = int(nodule_center[d] - (total_saved_patch_size[d] / 2))
                if not (int(nodule_center[d] + (total_saved_patch_size[d] / 2)) > scan_vol.shape[d]):
                    cropout_border[d, 1] = int(nodule_center[d] + (total_saved_patch_size[d] / 2))
            nodule_cropout_cube = scan_vol[cropout_border[0, 0]:cropout_border[0, 1],
                                  cropout_border[1, 0]:cropout_border[1, 1],
                                  cropout_border[2, 0]:cropout_border[2, 1]]
            nodulemask_cropout_cube = scan_segm[cropout_border[0, 0]:cropout_border[0, 1],
                                      cropout_border[1, 0]:cropout_border[1, 1],
                                      cropout_border[2, 0]:cropout_border[2, 1]]
            np.save("my_training_data/in_" + str(all_nodis_idx), nodule_cropout_cube)
            np.save("my_training_data/seg_" + str(all_nodis_idx), nodulemask_cropout_cube)
            all_nodis_idx += 1
            print("patch has size: " + str(nodule_cropout_cube.shape)+" from scan with size: "+str(scan_vol.shape))


def generateMyTestingData(args):
    scan_counter = 0
    for scan_folder in os.listdir(args.testing_data_path):
        all_patches_idx = 0
        print("Creating \"my_testing_data\" with custom preprocessed scan patches, at scan: " + str(
            scan_counter) + " of " + str(len(os.listdir(args.testing_data_path))))
        scan_vol = nib.load(args.testing_data_path + "/" + scan_folder + "/image_total.nii").get_fdata()

        os.makedirs("my_testing_data/" + scan_folder + "_" + str(scan_vol.shape))
        # Save scans patch wise
        firstdim_starts = torch.arange(0, scan_vol.shape[0], args.patch_size[0])
        seconddim_starts = torch.arange(0, scan_vol.shape[1], args.patch_size[1])
        thirddim_starts = torch.arange(0, scan_vol.shape[2], args.patch_size[2])
        for firstdim in firstdim_starts:
            for seconddim in seconddim_starts:
                for thirddim in thirddim_starts:
                    if (thirddim + args.patch_size[2]) >= scan_vol.shape[2]:
                        scan_vol_patch = np.zeros(args.patch_size)
                        scan_vol_patch[:, :, :(scan_vol.shape[2] - thirddim)] = \
                            scan_vol[firstdim:firstdim + args.patch_size[0],
                            seconddim:seconddim + args.patch_size[1],
                            thirddim:thirddim + args.patch_size[2]]
                    else:
                        scan_vol_patch = scan_vol[firstdim:firstdim + args.patch_size[0],
                                         seconddim:seconddim + args.patch_size[1],
                                         thirddim:thirddim + args.patch_size[2]]
                    np.save("my_testing_data/" + scan_folder + "_" + str(scan_vol.shape) + "/in_" + str(
                        all_patches_idx), scan_vol_patch)
                    all_patches_idx += 1
        scan_counter += 1
