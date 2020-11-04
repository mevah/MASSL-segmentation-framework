### Original code -> https://github.com/ShuaiChenBIGR/MASSL-segmentation-framework/blob/master/Data_BraTS2018.py
### By Shuai Chen, Erasmus MC

"""
This script needs to be run only once on the mrbrains parent directory
to produce the necessary resources for later use
This script is adapted from Brats_preprocess.py to be able to accomodate the needs of the paper
Instead of using all modalities, we only use T1 modality from mrbrains
"""

import numpy as np
from glob import glob
import SimpleITK as sitk
import matplotlib.pyplot as plt

from module.common_module import mkdir

import warnings
warnings.filterwarnings('ignore')

brats_main_dir = '/scratch_net/pengyou/himeva/data/training_corrected/training'
Patient_dir = sorted(glob(brats_main_dir + '/*'))
print(Patient_dir)
print('-' * 30)
print('Loading files...')
print('-' * 30)

num = 210
num_modality = 1

mean_all = 0
std_all = 0

for nb_file in range(len(Patient_dir)):

    # Set image path
    T1File = sorted(glob(Patient_dir[nb_file] + '/pre/reg_T1.nii.gz'))
    #T2File = sorted(glob(Patient_dir[nb_file] + "/Brats18_*_t2.nii.gz"))
    #FLAIRFile = sorted(glob(Patient_dir[nb_file] + "/Brats18_*_flair.nii.gz"))
    #T1cFile = sorted(glob(Patient_dir[nb_file] + "/Brats18_*_t1ce.nii.gz"))
    maskFile = sorted(glob(Patient_dir[nb_file] + "/segm.nii.gz"))
    # Read T1 image
    T1Image = sitk.ReadImage(T1File[0])
    T1Vol = sitk.GetArrayFromImage(T1Image).astype(float)
    T1mask = np.where(T1Vol >= 30, 1, 0)
    T1Vol = T1Vol * T1mask

    # Set padding and cut parameters
    cut_slice = int(0)

    size_h = 200
    size_w = 200

    if (T1Vol.shape[1] - size_h) % 2 == 0:
        crop_h1 = int((T1Vol.shape[1] - size_h)/2)
        crop_h2 = int((T1Vol.shape[1] - size_h)/2)
    else:
        crop_h1 = int((T1Vol.shape[1] - size_h)/2)
        crop_h2 = int((T1Vol.shape[1] - size_h)/2 + 1)

    if (T1Vol.shape[2] - size_w) % 2 == 0:
        crop_w1 = int((T1Vol.shape[2] - size_w)/2)
        crop_w2 = int((T1Vol.shape[2] - size_w)/2)
    else:
        crop_w1 = int((T1Vol.shape[2] - size_w)/2)
        crop_w2 = int((T1Vol.shape[2] - size_w)/2 + 1)

    # Cropping and cut image
    T1Vol = T1Vol[cut_slice:, crop_h1: (T1Vol.shape[1] - crop_h2), crop_w1:(T1Vol.shape[2] - crop_w2)]
    # Gaussion Normalization
    T1mask = np.where(T1Vol >= 30, 1, 0)
    mean = np.sum(T1Vol) / np.sum(T1mask)
    std = np.sqrt( np.sum((np.abs(T1Vol - mean)*T1mask)**2) / np.sum(T1mask) )
    T1Vol = T1Vol - mean
    T1Vol = T1Vol * T1mask
    T1Vol = T1Vol / std

    # Visualize image
    # plt.figure(figsize=(10, 6))
    # plt.suptitle("T1 image of patient {}".format(Patient_dir[nb_file].split('/')[-1]), fontsize=18)
    # start_slice = 75
    # for i in range(0, 6):
    #     T1slice = T1Vol[i * 20 + start_slice]
    #     ax = plt.subplot(2, 3, i + 1)
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     ax.set_title('Slice {}'.format(i + start_slice), fontsize=14)
    #     ax.axis('off')
    #     plt.imshow(T1slice, cmap='gray')
    #     plt.pause(0.001)
    # plt.show()
    # plt.close()

    # # Read T2 image
    # T2Image = sitk.ReadImage(T2File[0])
    # T2Vol = sitk.GetArrayFromImage(T2Image).astype(float)
    # T2mask = np.where(T2Vol >= 30, 1, 0)

    # T2Vol = T2Vol * T2mask

    # if (T2Vol.shape[1] - size_h) % 2 == 0:
    #     crop_h1 = int((T2Vol.shape[1] - size_h)/2)
    #     crop_h2 = int((T2Vol.shape[1] - size_h)/2)
    # else:
    #     crop_h1 = int((T2Vol.shape[1] - size_h)/2)
    #     crop_h2 = int((T2Vol.shape[1] - size_h)/2 + 1)

    # if (T2Vol.shape[2] - size_w) % 2 == 0:
    #     crop_w1 = int((T2Vol.shape[2] - size_w)/2)
    #     crop_w2 = int((T2Vol.shape[2] - size_w)/2)
    # else:
    #     crop_w1 = int((T2Vol.shape[2] - size_w)/2)
    #     crop_w2 = int((T2Vol.shape[2] - size_w)/2 + 1)

    # # Cropping and cut image
    # T2Vol = T2Vol[cut_slice:, crop_h1: (T2Vol.shape[1] - crop_h2), crop_w1:(T2Vol.shape[2] - crop_w2)]
    # # Gaussion Normalization
    # T2mask = np.where(T2Vol >= 30, 1, 0)
    # mean = np.sum(T2Vol) / np.sum(T2mask)
    # std = np.sqrt( np.sum((np.abs(T2Vol - mean)*T2mask)**2) / np.sum(T2mask) )
    # T2Vol = T2Vol - mean
    # T2Vol = T2Vol * T2mask
    # T2Vol = T2Vol / std

    # #  Read FLAIR image
    # FLAIRImage = sitk.ReadImage(FLAIRFile[0])
    # FLAIRVol = sitk.GetArrayFromImage(FLAIRImage).astype(float)
    # FLAIRmask = np.where(FLAIRVol >= 30, 1, 0)

    # FLAIRVol = FLAIRVol * FLAIRmask

    # if (FLAIRVol.shape[1] - size_h) % 2 == 0:
    #     crop_h1 = int((FLAIRVol.shape[1] - size_h)/2)
    #     crop_h2 = int((FLAIRVol.shape[1] - size_h)/2)
    # else:
    #     crop_h1 = int((FLAIRVol.shape[1] - size_h)/2)
    #     crop_h2 = int((FLAIRVol.shape[1] - size_h)/2 + 1)

    # if (FLAIRVol.shape[2] - size_w) % 2 == 0:
    #     crop_w1 = int((FLAIRVol.shape[2] - size_w)/2)
    #     crop_w2 = int((FLAIRVol.shape[2] - size_w)/2)
    # else:
    #     crop_w1 = int((FLAIRVol.shape[2] - size_w)/2)
    #     crop_w2 = int((FLAIRVol.shape[2] - size_w)/2 + 1)

    # # Cropping and cut image
    # FLAIRVol = FLAIRVol[cut_slice:, crop_h1: (FLAIRVol.shape[1] - crop_h2), crop_w1:(FLAIRVol.shape[2] - crop_w2)]
    # # Gaussion Normalization
    # FLAIRmask = np.where(FLAIRVol >= 30, 1, 0)
    # mean = np.sum(FLAIRVol) / np.sum(FLAIRmask)
    # std = np.sqrt( np.sum((np.abs(FLAIRVol - mean)*FLAIRmask)**2) / np.sum(FLAIRmask) )
    # FLAIRVol = FLAIRVol - mean
    # FLAIRVol = FLAIRVol * FLAIRmask
    # FLAIRVol = FLAIRVol / std

    # mean_all += mean
    # std_all += std
    # print('mean', mean)
    # print('std', std)

    # #  Read T1c image
    # T1cImage = sitk.ReadImage(T1cFile[0])
    # T1cVol = sitk.GetArrayFromImage(T1cImage).astype(float)
    # T1cmask = np.where(T1cVol >= 30, 1, 0)

    # T1cVol = T1cVol * T1cmask

    # if (T1cVol.shape[1] - size_h) % 2 == 0:
    #     crop_h1 = int((T1cVol.shape[1] - size_h)/2)
    #     crop_h2 = int((T1cVol.shape[1] - size_h)/2)
    # else:
    #     crop_h1 = int((T1cVol.shape[1] - size_h)/2)
    #     crop_h2 = int((T1cVol.shape[1] - size_h)/2 + 1)

    # if (T1cVol.shape[2] - size_w) % 2 == 0:
    #     crop_w1 = int((T1cVol.shape[2] - size_w)/2)
    #     crop_w2 = int((T1cVol.shape[2] - size_w)/2)
    # else:
    #     crop_w1 = int((T1cVol.shape[2] - size_w)/2)
    #     crop_w2 = int((T1cVol.shape[2] - size_w)/2 + 1)

    # # Cropping and cut image
    # T1cVol = T1cVol[cut_slice:, crop_h1: (T1cVol.shape[1] - crop_h2), crop_w1:(T1cVol.shape[2] - crop_w2)]
    # # Gaussion Normalization
    # T1cmask = np.where(T1cVol >= 30, 1, 0)
    # mean = np.sum(T1cVol) / np.sum(T1cmask)
    # std = np.sqrt( np.sum((np.abs(T1cVol - mean)*T1cmask)**2) / np.sum(T1cmask) )
    # T1cVol = T1cVol - mean
    # T1cVol = T1cVol * T1cmask
    # T1cVol = T1cVol / std

    # Read mask file
    maskImage = sitk.ReadImage(maskFile[0])
    maskVol = sitk.GetArrayFromImage(maskImage).astype(float)
    # print(np.unique(maskVol))

    maskVol = np.where(maskVol == 5, 0, maskVol)
    maskVol = np.where(maskVol == 6, 5, maskVol)
    maskVol = np.where(maskVol == 7, 6, maskVol)
    maskVol = np.where(maskVol == 8, 7, maskVol)

    maskVol = np.where(maskVol == 10, 1, maskVol)
    maskVol = np.where(maskVol == 9, 4, maskVol)

    # print(np.unique(maskVol))

    # Padding and cut image
    maskVol = maskVol[cut_slice:, crop_h1: (maskVol.shape[1] - crop_h2), crop_w1:(maskVol.shape[2] - crop_w2)]
 #   imageVol = np.concatenate((np.expand_dims(T1Vol, axis=0), np.expand_dims(T2Vol, axis=0), np.expand_dims(FLAIRVol, axis=0), np.expand_dims(T1cVol, axis=0)), axis=0)
    imageVol =np.expand_dims(T1Vol, axis=0)
    print(str(Patient_dir[nb_file].split('/')[0]))
    mkdir('/scratch_net/pengyou/himeva/data/MRBrains18')
    np.save('/scratch_net/pengyou/himeva/data/MRBrains18' + '/img_%s.npy' % (str(Patient_dir[nb_file].split('/training/')[-1][:3])), imageVol)
    np.save('/scratch_net/pengyou/himeva/data/MRBrains18' + '/mask_%s.npy' % (str(Patient_dir[nb_file].split('/training/')[-1][:3])), maskVol)

    print('MRBrains18 Image process {}/{} finished'.format(nb_file, len(Patient_dir)))


print('finished')

mean_all /= 210
std_all /= 210

print(mean_all, std_all)

