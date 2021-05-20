import SimpleITK as sitk
import numpy as np
import glob
from os.path import splitext, split

imgs_dir = './data/imgs/'
def histogram_matching():
    # Load the template image
    # template = nib.load(reference_nii)
    img_dir = glob.glob(imgs_dir+'*.nii')
    img_dir.remove('./data/imgs/1.nii')
    for dir in img_dir:
        template = sitk.ReadImage('./data/imgs/1.nii')
        nt_data = sitk.GetArrayFromImage(template)

        # Load the patient image
        # patient = nib.load(input_nii)
        patient = sitk.ReadImage(dir)
        pt_data = sitk.GetArrayFromImage(patient)

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
        out_img = sitk.GetImageFromArray(out_img.astype(np.int16))
        sitk.WriteImage(out_img, imgs_dir + 'hist_newImg/{}'.format(split(dir)[1]))
        # final_image_data[indx] = 0

        # Saves the output data
        # img = nib.Nifti1Image(final_image_data, patient.affine, patient.header)
        # nib.save(img, output_nii)

his_match = histogram_matching()
