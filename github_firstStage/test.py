import SimpleITK as sitk
import numpy as np
from skimage import transform
import cv2
import torch

x = torch.rand(5,3)
y = torch.rand(5,3)
print(x+y)
print(torch.add(x,y))

result = torch.empty(5,3)
torch.add(x,y, out=result)
print(result)
y.add_(x)
print(y)

def histogram_matching(reference_nii, input_nii):
    # Load the template image
    # template = nib.load(reference_nii)
    nt_data = reference_nii[:, :, :]

    # Load the patient image
    # patient = nib.load(input_nii)
    pt_data = input_nii[:, :, :]

    # Stores the image data shape that will be used later
    oldshape = pt_data.shape

    # Converts the data arrays to single dimension and normalizes by the maximum
    nt_data_array = nt_data.ravel()
    pt_data_array = pt_data.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(pt_data_array, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(nt_data_array, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    # Reshapes the corresponding values to the indexes and reshapes the array to input
    out_img = interp_t_values[bin_idx].reshape(oldshape)
    # final_image_data[indx] = 0

    # Saves the output data
    # img = nib.Nifti1Image(final_image_data, patient.affine, patient.header)
    # nib.save(img, output_nii)

    return out_img

mask_file = './data/masks/002gt.nii'
#img_file = './data/imgs/cbct_hist.nii'
mask = sitk.ReadImage(mask_file)
mask = sitk.GetArrayFromImage(mask)
#img = sitk.ReadImage(img_file)
#img = sitk.GetArrayFromImage(img)
img_1 = transform.resize(mask, (96,96,96),preserve_range=False)
a = img_1*65536
img_nd = np.array(img_1)

if len(img_nd.shape) == 3:
    img_nd = np.expand_dims(img_nd, axis=3)

# HWC to CHW
img_trans = img_nd.transpose((3, 0, 1, 2))
if img_trans.max() > 1:
    img_trans = img_trans / 255
out_img = histogram_matching(mask, img)
out_img = sitk.GetImageFromArray(out_img)
sitk.WriteImage(out_img, './data/imgs/cbct_hist.nii')
# img_nd = np.array(img_1)
# # img_nd = np.expand_dims(img_nd, axis=3)
# # img_trans = img_nd.transpose((3, 0, 1, 1))
# # print(img_trans)

