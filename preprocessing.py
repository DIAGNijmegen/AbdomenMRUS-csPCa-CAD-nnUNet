import os
import SimpleITK as sitk
import numpy as np
from skimage.measure import regionprops

try:
    import cv2
except ImportError:
    print("Importing opencv failed, functions using it will fail")

from data_utils import atomic_image_write

'''
Binary PCa Detection in mpMRI
Script:         Preprocessing of mpMRI scans
Contributor:    anindox8, joeranbosma
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
'''


# Resample Images to Target Resolution Spacing [Ref:SimpleITK]
def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False, pad_value='auto'):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    if pad_value == 'auto':
        resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    else:
        resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    # perform resampling
    itk_image = resample.Execute(itk_image)

    return itk_image


# Center Crop NumPy Volumes
def center_crop(img, cropz=None, cropx=None, cropy=None, center_2d_coords=None):
    if cropz is None:
        cropz = img.shape[0]
    if cropx is None:
        cropx = img.shape[2]
    if cropy is None:
        cropy = img.shape[1]
    if center_2d_coords:
        x, y = center_2d_coords
    else:
        x, y = img.shape[2]//2, img.shape[1]//2

    startz = img.shape[0]//2 - (cropz//2)
    startx = int(x) - (cropx//2)
    starty = int(y) - (cropy//2)

    if not (0 <= startz <= img.shape[0] and 0 <= startx <= img.shape[2] and 0 <= starty <= img.shape[1]):
        # Obtained invalid crop size, which can yield unexpected behaviour
        # Pad image until large enough size to allow proper centre crop
        img_size = (
            cropz,
            y+cropy//2+1,
            x+cropx//2+1,
        )
        img = resize_image_with_crop_or_pad(img, img_size=img_size)
        return center_crop(img, cropz=cropz, cropx=cropx, cropy=cropy, center_2d_coords=center_2d_coords)

    return img[startz:startz+cropz, starty:starty+cropy, startx:startx+cropx]


# Resize Image with Crop/Pad [Ref:DLTK]
def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    rank = len(img_size)  # Image Dimensions

    # Placeholders for New Shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]
    slicer = [slice(None)] * rank

    # For Each Dimension Determine Process (Cropping/Padding)
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]
        # Create Slicer Object to Crop/Leave Each Dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad Cropped Image to Extend Missing Dimension
    return np.pad(image[tuple(slicer)], to_padding, **kwargs)


def size_to_dim(size, res, assert_multiple=False):
    """
    Convert physical size to number of voxels, with given resolution in mm/voxel.
    - assert_multiple: yield an error if the number of voxels is not a round number
    """
    dim = size / res
    if assert_multiple:
        assert dim % 1 == 0, f"Converting {size}mm with resolution {res} yielded non-integer number of voxels {dim}!"
    return int(dim)


def compute_lcm(x, y):
    """Calculate the least common divisor of two integers. """
    # choose the greater number
    if x > y:
        greater = x
    else:
        greater = y

    while True:
        if (greater % x == 0) and (greater % y == 0):
            return greater
        greater += 1


def get_overlap_start_indices(img_T2W, img_ADC):
    # convert start index from ADC to T2W
    point_ADC = img_ADC.TransformIndexToPhysicalPoint((0, 0, 0))
    index_T2W = img_T2W.TransformPhysicalPointToContinuousIndex(point_ADC)

    # clip index
    index_T2W = np.clip(index_T2W, a_min=0, a_max=None)

    # convert T2W index back to ADC
    point_T2W = img_T2W.TransformContinuousIndexToPhysicalPoint(index_T2W)
    index_ADC = img_ADC.TransformPhysicalPointToContinuousIndex(point_T2W)

    # round ADC index up
    index_ADC = np.ceil(np.round(index_ADC, decimals=5))

    # convert ADC index once again to T2W
    point_ADC = img_ADC.TransformContinuousIndexToPhysicalPoint(index_ADC)
    index_T2W = img_T2W.TransformPhysicalPointToIndex(point_ADC)

    return np.array(index_ADC).astype(int), np.array(index_T2W).astype(int)


def get_overlap_end_indices(img_T2W, img_ADC):
    # convert end index from ADC to T2W
    point_ADC = img_ADC.TransformIndexToPhysicalPoint(img_ADC.GetSize())
    index_T2W = img_T2W.TransformPhysicalPointToContinuousIndex(point_ADC)

    # clip index
    index_T2W = [min(sz, i) for (i, sz) in zip(index_T2W, img_T2W.GetSize())]

    # convert T2W index back to ADC
    point_T2W = img_T2W.TransformContinuousIndexToPhysicalPoint(index_T2W)
    index_ADC = img_ADC.TransformPhysicalPointToContinuousIndex(point_T2W)

    # round ADC index down
    index_ADC = np.floor(np.round(index_ADC, decimals=5))  # first round to 5 decimals for e.g. 18.999999999999996

    # convert ADC index once again to T2W
    point_ADC = img_ADC.TransformContinuousIndexToPhysicalPoint(index_ADC)
    index_T2W = img_T2W.TransformPhysicalPointToIndex(point_ADC)

    return np.array(index_ADC).astype(int), np.array(index_T2W).astype(int)


def crop_to_common_physical_space(sample, key_main='img_T2W', key_sec='img_ADC',
                                  shared_keys_main='auto', shared_keys_sec='auto'):
    """
    Crops the SimpleITK images to the largest shared physical volume
    - key_main and key_sec: names of the scans in sample['features'] to use to compare the physical space
    - shared_keys_main: names of the scans in sample['features'] that share the physical space of key_main
    - shared_keys_sec: names of the scans in sample['features'] that share the physical space of key_sec
    (e.g., DWI & ADC & label share the physical space, and so do T2 & zonal mask)
    """
    if shared_keys_main == 'auto':
        if key_main == 'img_T2W':
            shared_keys_main = [key_main, 'zonal_mask']
        else:
            shared_keys_main = [key_main]
    if shared_keys_sec == 'auto':
        if key_sec == 'img_ADC':
            shared_keys_sec = [key_sec, 'img_DWI', 'lbl']
        else:
            shared_keys_sec = [key_sec]

    # grab main scan (T2W) and secondary scan (ADC) image to calculate overlap in physical space
    img_main = sample['features'][key_main]
    img_sec = sample['features'][key_sec]

    # determine start indices
    idx_start_sec, idx_start_main = get_overlap_start_indices(img_main, img_sec)
    idx_end_sec, idx_end_main = get_overlap_end_indices(img_main, img_sec)

    # check extracted indices
    assert ((idx_end_sec - idx_start_sec) > np.array(img_sec.GetSize()) / 2).all(), \
        f"Found unrealistically little overlap between {key_main} and {key_sec}, aborting."
    assert ((idx_end_main - idx_start_main) > np.array(img_main.GetSize()) / 2).all(), \
        f"Found unrealistically little overlap between {key_main} and {key_sec}, aborting."

    for section in ['features', 'labels']:
        if shared_keys_main is not None:
            for key in shared_keys_main:
                slices = [slice(idx_start, idx_end) for (idx_start, idx_end) in zip(idx_start_main, idx_end_main)]
                if key in sample[section]:
                    sample[section][key] = sample[section][key][slices]

    for section in ['features', 'labels']:
        if shared_keys_sec is not None:
            for key in shared_keys_sec:
                slices = [slice(idx_start, idx_end) for (idx_start, idx_end) in zip(idx_start_sec, idx_end_sec)]
                if key in sample[section]:
                    sample[section][key] = sample[section][key][slices]
    return sample


def preprocess_scans_mpMRI_study(sample, physical_size=(3.6*18, 0.5*144, 0.5*144), center_prostate=False,
                                 center_min_prostate_volume=1, center_max_prostate_volume=None,
                                 align_physical_space=False, resample_uniform_spacing=None,
                                 main_centre=None, subject_id=None, verbose=0):
    """
    Preprocess scans.

    Resample to the spacing listed below:
         |  x & y                       |  z
    -------------------------------------------
    T2w  |  0.3 --> 0.3, other --> 0.5  | 3.6
    ADC  |  2.0                         | 3.6
    HBV  |  2.0                         | 3.6
    lbl  |  2.0                         | 3.6
    zonal|  0.3 --> 0.3, other --> 0.5  | 3.6

    - physical_size: size in mm/voxel of the target volume.
    - center_prostate: whether to center the scans using the centroid of the prostate segmentation
    - center_min_volume_prostate: minimum volume of the prostate segmentation in order to use it.
    - center_max_volume_prostate: maximum volume of the prostate segmentation in order to use it.
    Setting the min and max prostate volume to e.g. the 5th and 95th percentiles increases the chance the
        segmentation is correct.

    The sequences T2W, ADC and DWI are assumed to be present always. The lbl and zonal_mask are optional.
    """

    # grab images
    all_scans = list(sample['features'].values())
    lbl = sample['labels']['lbl'] if 'lbl' in sample['labels'] else None

    if lbl is not None:
        # check if label has a malignancy to check correct behaviour of this function
        malignant_start = (sitk.GetArrayFromImage(lbl).sum() > 0)

    if align_physical_space:
        # compare physical centers of the first scan (T2W) and secondary scans (ADC/high b-value/DCE).
        # The ADC and DWI scans are always the same physical space.
        main_center = all_scans[0].TransformContinuousIndexToPhysicalPoint(np.array(all_scans[0].GetSize())/2.0)
        should_align_scans = False
        for scan in all_scans[1:]:
            secondary_center = scan.TransformContinuousIndexToPhysicalPoint(np.array(scan.GetSize())/2.0)

            # calculate distance from center of first scan (T2W) to center of secondary scan (ADC/high b-value/DCE).
            distance = np.sqrt(np.sum((np.array(main_center) - np.array(secondary_center))**2))
            # if difference in center coordinates is more than 2mm, align the scans
            if distance > 2:
                print(f"Aligning scans with distance of {distance:.1f} mm between centers for {subject_id}.")
                should_align_scans = True

        if should_align_scans:
            for key_main in sample['features']:
                for key_sec in sample['features']:
                    if key_main == key_sec:
                        continue

                    # align scans
                    sample = crop_to_common_physical_space(sample, key_main=key_main, key_sec=key_sec)

            # grab images
            all_scans = list(sample['features'].values())
            lbl = sample['labels']['lbl'] if 'lbl' in sample['labels'] else None

    # resample scans
    if resample_uniform_spacing is not None:
        # uniform spacing (e.g. for nnUNet)
        res_main = resample_uniform_spacing
        res_sec = resample_uniform_spacing
    else:
        # dynamic spacing
        res_main = (0.3, 0.3, 3.6) if 0.3 == round(all_scans[0].GetSpacing()[0], 1) else (0.5, 0.5, 3.6)
        res_sec = (2.0, 2.0, 3.6)

    all_res = [res_main]+[res_sec]*(len(all_scans)-1)

    # resample scans and label
    all_scans[0] = resample_img(all_scans[0], out_spacing=res_main, is_label=False, pad_value='auto')
    all_scans[1:] = [resample_img(img, out_spacing=res_sec, is_label=False, pad_value='auto')
                     for img in all_scans[1:]]
    if lbl is not None:
        lbl = resample_img(lbl, out_spacing=res_sec, is_label=True, pad_value='auto')

    if main_centre is not None:
        # Crop images to Physical Centre of main image
        for i, scan in enumerate(all_scans):
            # determine indices of physical centre
            indexes = scan.TransformPhysicalPointToIndex(main_centre)

            # calculate which start and end indices crop the image to
            # the largest image around the physical centre
            half_size = [min(indexes[i], scan.GetSize()[i] - indexes[i])
                         for i in range(3)]
            # apply crop
            slices = [slice(idx_centre-half, idx_centre+half)
                      for idx_centre, half in zip(indexes, half_size)]
            scan = scan[slices]

            # save crop
            all_scans[i] = scan

    # transform images to numpy
    all_scans = [sitk.GetArrayFromImage(img) for img in all_scans]
    if lbl is not None:
        lbl = sitk.GetArrayFromImage(lbl)

    # Center Crop ADC, DWI, Annotation Scans to Same Scope
    zdim = min([img.shape[0] for img in all_scans])
    xysize = min([img.shape[1]*res[0] for img, res in zip(all_scans, all_res)] +
                 [img.shape[2]*res[1] for img, res in zip(all_scans, all_res)])
    if lbl is not None:
        zdim = min(zdim, lbl.shape[0])
        xysize = min(xysize, lbl.shape[1]*res_sec[0], lbl.shape[2]*res_sec[1])

    try:
        xydim_main = size_to_dim(xysize, res_main[0], assert_multiple=True)
        xydim_sec = size_to_dim(xysize, res_sec[0],  assert_multiple=True)
    except AssertionError:
        # for an example, see 1215911_96818115
        print(f"Reducing center crop size of {xysize}mm by {xysize} mm to have an integer number of voxels for each scan")
        eps = 1e-12
        assert res_main[0]*10 % 1 < eps and res_sec[0] * 10 % 1 < eps, "Failed to convert resolution to integers"
        common_res = compute_lcm(int(res_main[0]*10), int(res_sec[0]*10)) / 10
        xysize -= xysize % common_res
        xydim_main = size_to_dim(xysize, res_main[0], assert_multiple=True)
        xydim_sec = size_to_dim(xysize, res_sec[0],  assert_multiple=True)
        print(f"New center crop size: {xysize}mm by {xysize} mm.")

    all_xydim = [xydim_main] + [xydim_sec]*(len(all_scans) - 1)

    # apply centre crop
    for i, (scan, xydim) in enumerate(zip(all_scans, all_xydim)):
        all_scans[i] = center_crop(scan, zdim, xydim, xydim)

    if lbl is not None:
        lbl = center_crop(lbl, zdim, xydim_sec, xydim_sec)

    # Preprocess and Clean Labels (from Possible Border Interpolation Errors)
    # Deprecated, can have granular annotations: lbl[(lbl!=0)&(lbl!=1)]    = 0 if lbl is not None else None

    # Padding if Z-Dimension is Below Minimum Center-Crop Dimension
    crop_z_dims = size_to_dim(physical_size[0], res_main[2], assert_multiple=True)
    if any([scan.shape[0] < crop_z_dims for scan in all_scans]) or (lbl is not None and lbl.shape[0] < crop_z_dims):
        all_scans = [
            resize_image_with_crop_or_pad(scan, img_size=(crop_z_dims, scan.shape[1], scan.shape[2]))
            for scan in all_scans
        ]
        if lbl is not None:
            lbl = resize_image_with_crop_or_pad(lbl, img_size=(crop_z_dims, lbl.shape[1], lbl.shape[2]))

    center_coords_main, center_coords_sec = None, None
    if center_prostate:
        # calculate number of voxel inside prostate
        zonal_mask = sample['features']['zonal_mask']
        nrOfVoxelsInMask = (zonal_mask > 0).sum()  # number of voxels in prostate

        # calculate volume
        spacing = res_main  # spacing is mm/voxel
        voxel_volume = spacing[0] * spacing[1] * spacing[2]  # volume is mm^3 / voxel
        total_volume = nrOfVoxelsInMask * voxel_volume / 1000  # volume in cc (cm^3)

        # convert a center_max_prostate_volume of -1 to None
        if center_max_prostate_volume == -1:
            center_max_prostate_volume = None

        if total_volume > center_min_prostate_volume and (center_max_prostate_volume is None or total_volume < center_max_prostate_volume):
            bin_mask = (zonal_mask > 0).astype(int)
            # max_area_slice = np.argmax( np.sum(np.sum(bin_mask, axis=-1), axis=-1) )
            # center_coords_T2W  = regionprops(bin_mask[max_area_slice])[0].centroid
            centroid = regionprops(bin_mask)[0].centroid  # returns tuple of (z, y, x)
            center_coords_main = (centroid[2], centroid[1])
            center_coords_sec = [(v * res_main[i] / res_sec[i]) for i, v in enumerate(center_coords_main)]

            print(f"Centring scans with centroid of zonal segmentation: {center_coords_main} or {center_coords_sec}") if verbose >= 2 else None
        else:
            print("Prostate segmentation not within min/max volume for centring scan") if verbose >= 2 else None

    # Center Crop Volumes to ROI with Same Volume
    crop_xy_dims_main = size_to_dim(physical_size[1], res_main[1], assert_multiple=True)
    crop_xy_dims_sec = size_to_dim(physical_size[1], res_sec[1], assert_multiple=True)
    all_crop_xy_dims = [crop_xy_dims_main]+[crop_xy_dims_sec]*(len(all_scans)-1)
    all_center_coords = [center_coords_main]+[center_coords_sec]*(len(all_scans)-1)
    all_scans = [
        center_crop(scan, crop_z_dims, crop_xy_dims, crop_xy_dims, center_2d_coords=center_coords)
        for scan, crop_xy_dims, center_coords in zip(all_scans, all_crop_xy_dims, all_center_coords)
    ]
    if lbl is not None:
        lbl = center_crop(lbl, crop_z_dims, crop_xy_dims_sec, crop_xy_dims_sec, center_2d_coords=center_coords_sec)

    if lbl is not None:
        malignant_end = (lbl.sum() > 0)
        assert malignant_start == malignant_end, "Label has changed due to interpolation/other errors!"

    # pack sample
    for key, scan in zip(sample['features'], all_scans):
        sample['features'][key] = scan
    if lbl is not None:
        sample['labels']['lbl'] = lbl

    return sample


def translate_pred_to_reference_scan(pred: np.array,
                                     reference_scan: sitk.Image,
                                     out_spacing: tuple = (0.5, 0.5, 3.6),
                                     is_label: bool = False) -> sitk.Image:
    """
    Translate prediction back to physical space of input T2 scan
    This function performs the reverse operation of the preprocess_study function
    - pred: softmax / binary prediction
    - reference_scan: SimpleITK image to which the prediction should be resampled and resized
    - out_spacing: spacing to which the reference scan is resampled during preprocessing
    """
    reference_scan_resampled = resample_img(reference_scan, out_spacing=out_spacing, is_label=False, pad_value=0)

    # pad softmax prediction to physical size of resampled reference scan (with inverted order of image sizes)
    pred = resize_image_with_crop_or_pad(pred, img_size=list(reference_scan_resampled.GetSize())[::-1])
    pred_itk = sitk.GetImageFromArray(pred)

    # set the physical properties of the predictions
    pred_itk.CopyInformation(reference_scan_resampled)

    # resample predictions to spacing of original reference scan
    pred_itk_resampled = resample_img(pred_itk, out_spacing=reference_scan.GetSpacing(), is_label=is_label, pad_value=0)
    return pred_itk_resampled


def read_itk_with_header(path, dtype):
    meta_data = dict()
    itk_img = sitk.ReadImage(path, dtype)

    meta_data['itk_size'] = list(itk_img.GetSize())
    meta_data['itk_spacing'] = list(itk_img.GetSpacing())
    meta_data['itk_origin'] = list(itk_img.GetOrigin())
    meta_data['itk_direction'] = list(itk_img.GetDirection())
    return itk_img, meta_data


def write_itk_with_header(np_img, path, meta_data):
    itk_img = sitk.GetImageFromArray(np_img)
    itk_img.SetOrigin(meta_data['itk_origin'])
    itk_img.SetSpacing((0.5, 0.5, 3.6))
    itk_img.SetDirection(meta_data['itk_direction'])

    # write image with failsafe for incomplete write actions
    atomic_image_write(itk_img, path, extension='.nii.gz')


def grab_dtype(properties, dtype='float32'):
    if 'dtype' in properties:
        dtype = properties['dtype']
    if dtype == 'float32':
        dtype = sitk.sitkFloat32
    if dtype == 'uint8':
        dtype = sitk.sitkUInt8
    return dtype


def preprocess_mpMRI_study(all_scan_properties, label_properties=None, subject_id=None, physical_size=None,
                           resample_uniform_spacing=None, align_physical_space=True,
                           crop_to_first_physical_centre=False, apply_binary_smoothing=False,
                           overwrite_files=True, dummy_ktrans_image=False, dry_run=False):
    """
    Preprocess studies for nnUNet
    - all_scan_properties: list of scan properties:
    [
        {
            'input_path': path to input scan (T2/ADC/high b-value/etc.),
            'output_path': path to store preprocessed scan,
            'dtype': SimpleITK dtype (default is sitk.sitkFloat32),
        }
    ]

    - label_properties: label properties:
    {
        'input_path': path to label (at ADC resolution),
        'output_path': path to store preprocessed label,
        'dtype': SimpleITK dtype (default is sitk.sitkUInt8),
    }

    - dummy_ktrans_image: whether the Ktrans image is a dummy (because the DCE images are missing/cannot be aligned)
    """
    if physical_size is None:
        physical_size = (72.0, 80.0, 80.0)
    if resample_uniform_spacing is None:
        resample_uniform_spacing = (0.5, 0.5, 3.6)

    # read images
    all_scans = []
    all_metadata = []
    for scan_properties in all_scan_properties:
        dtype = grab_dtype(scan_properties)
        img, meta = read_itk_with_header(scan_properties['input_path'], dtype=dtype)
        all_scans += [img]
        all_metadata += [meta]

    if crop_to_first_physical_centre:
        # determine main centre
        scan = all_scans[0]
        main_centre = scan.TransformContinuousIndexToPhysicalPoint(np.array(scan.GetSize()) / 2)
    else:
        main_centre = None

    # read label
    lbl = None
    if label_properties is not None:
        if "dummy_from_" in label_properties['input_path']:
            # should make a dummy label
            from_modality = label_properties['input_path'].split("dummy_from_")[1]
            scan_properties = [scan_properties for scan_properties in all_scan_properties
                               if scan_properties['type'] == from_modality][0]

            # read reference scan
            scan = sitk.ReadImage(scan_properties['input_path'])

            # create dummy label
            lbl = np.zeros_like(sitk.GetArrayFromImage(scan))
            lbl = sitk.GetImageFromArray(lbl)
            lbl.CopyInformation(scan)
        else:
            dtype = grab_dtype(label_properties, dtype='uint8')
            lbl, meta = read_itk_with_header(label_properties['input_path'], dtype=dtype)

    if lbl is not None:
        if 'type' not in label_properties:
            raise ValueError("Label type must be specified explicitly!")

        label_type = label_properties['type']
        mask = sitk.GetArrayFromImage(lbl)
        if label_type == 'zonal_segmentation_to_whole_gland_segmentation':
            # convert zonal segmentation to whole gland segmentation
            mask = (mask > 0)
            mask = mask.astype('uint8')
        elif label_type == 'granular_pirads':
            # convert granular annotation to csPCa annotation
            mask = (mask == 1) | (mask == 4) | (mask == 5)
            mask = mask.astype('uint8')
        elif label_type == 'binary_pirads':
            # 'convert' binary annotation to csPCa annotation
            assert np.unique(mask).sum() <= 1, \
                f"Invalid value encountered in binary PI-RADS annotation: {np.unique(mask)}" + \
                f" (in {label_properties['input_path']}"
            mask = (mask == 1).astype('uint8')
        elif label_type == 'granular_pathology':
            # convert granular pathology annotation to csPCa annotation
            mask = (mask >= 2).astype('uint8')
        elif label_type == 'binary_pathology':
            # 'convert' binary pathology annotation to csPCa annotation
            assert np.unique(mask).sum() <= 1, \
                f"Invalid value encountered in binary pathology annotation: {np.unique(mask)}" + \
                f" (in {label_properties['input_path']}"
            mask = (mask == 1).astype('uint8')
        elif label_type == 'dummy':
            # dummy label, nothing to do
            pass
        else:
            raise ValueError(f"Unexpected label type: {label_type}")

        # keep track of whether case is positive
        is_malignant = (mask.sum() > 0)

        # convert binary label to SimpleITK
        new_lbl = sitk.GetImageFromArray(mask)
        new_lbl.CopyInformation(lbl)
        lbl = new_lbl

    # apply preprocessing: resample and centre crop
    sample = {'features': {
        f"{i:04d}": scan for i, scan in enumerate(all_scans)
    }, 'labels': {}}
    if lbl is not None:
        sample['labels']['lbl'] = lbl
    sample = preprocess_scans_mpMRI_study(sample, align_physical_space=align_physical_space, physical_size=physical_size,
                                          resample_uniform_spacing=resample_uniform_spacing, main_centre=main_centre,
                                          subject_id=subject_id)
    all_scans = list(sample['features'].values())
    if lbl is not None:
        lbl = sample['labels']['lbl']

    if lbl is not None:
        if apply_binary_smoothing:
            # apply binary label smoothing
            for i in range(lbl.shape[0]):
                lbl[i, :, :] = cv2.GaussianBlur(lbl[i, :, :], (9, 9), cv2.BORDER_DEFAULT)

        lbl = (lbl > 0.5).astype('uint8')
        if is_malignant and not lbl.sum() > 0:
            raise AssertionError(f"Label switched value due to preprocessing or binary label smoothing for {subject_id}!")

    if dry_run:
        # skip saving images if dry run
        print(f"Skipped saving images & labels for {subject_id}")
        return all_scans

    # write images
    for scan, scan_properties in zip(all_scans, all_scan_properties):
        if overwrite_files or not os.path.exists(scan_properties['output_path']):
            if dummy_ktrans_image and 'type' in scan_properties and scan_properties['type'] == 'Ktrans':
                # set dummy Ktrans image to zero
                scan = (scan * 0)

            write_itk_with_header(scan, scan_properties['output_path'], all_metadata[0])

    if lbl is not None:
        if overwrite_files or not os.path.exists(label_properties['output_path']):
            write_itk_with_header(lbl, label_properties['output_path'], all_metadata[0])

    return 1
