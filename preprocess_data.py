from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
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
Script:         Preprocessing
Contributor:    anindox8, joeranbosma
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
'''


# Resample Images to Target Resolution Spacing [Ref:SimpleITK]
def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size    = itk_image.GetSize()
    
    out_size = [ int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                 int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                 int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    
    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    # perform resampling
    itk_image = resample.Execute(itk_image)

    return itk_image


# Center Crop NumPy Volumes
def center_crop(img, cropz=None, cropx=None, cropy=None, center_2d_coords=None, multi_channel=False):
    if cropz is None: cropz  = img.shape[0]
    if cropx is None: cropx  = img.shape[2]
    if cropy is None: cropy  = img.shape[1]
    if center_2d_coords: x,y = center_2d_coords
    else:                x,y = img.shape[2]//2, img.shape[1]//2
  
    startz = img.shape[0]//2 - (cropz//2)
    startx = int(x) - (cropx//2)
    starty = int(y) - (cropy//2)
    
    assert 0 <= startz <= img.shape[0] and 0 <= startx <= img.shape[2] and 0 <= starty <= img.shape[1], \
        f"Obtained invalid crop size which can yield unexpected behaviour. Got (x, y, z) = {startx, starty, startz}"
    
    if (multi_channel==True): return img[startz:startz+cropz,starty:starty+cropy,startx:startx+cropx,:]
    else:                     return img[startz:startz+cropz,starty:starty+cropy,startx:startx+cropx]


# Resize Image with Crop/Pad [Ref:DLTK]
def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    rank = len(img_size)  # Image Dimensions

    # Placeholders for New Shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding   = [[0, 0] for dim in range(rank)]
    slicer       = [slice(None)] * rank

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
    if x > y: greater = x
    else:     greater = y
    
    while True:
        if (greater % x == 0) and (greater % y == 0):
            return greater
        greater += 1


def get_overlap_start_indices(img_T2W, img_ADC):
    # convert start index from ADC to T2W
    point_ADC = img_ADC.TransformIndexToPhysicalPoint((0,0,0))
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
    index_ADC = np.floor(np.round(index_ADC, decimals=5)) # first round to 5 decimals for e.g. 18.999999999999996
    
    # convert ADC index once again to T2W
    point_ADC = img_ADC.TransformContinuousIndexToPhysicalPoint(index_ADC)
    index_T2W = img_T2W.TransformPhysicalPointToIndex(point_ADC)
    
    return np.array(index_ADC).astype(int), np.array(index_T2W).astype(int)


def crop_to_common_physical_space(sample):
    """
    Crops the SimpleITK images to the largest shared physical volume
    """
    # convert direction to RAI
#     for img in sample['features'].values():
#         img.SetDirection(np.identity(3).flatten())
    
    # grab T2W and ADC image to calculate overlap in physical space
    img_T2W = sample['features']['img_T2W']
    img_ADC = sample['features']['img_ADC']
    
    # determine start indices
    idx_start_ADC, idx_start_T2W = get_overlap_start_indices(img_T2W, img_ADC)
    idx_end_ADC, idx_end_T2W = get_overlap_end_indices(img_T2W, img_ADC)

    # check extracted indices
    assert ((idx_end_ADC - idx_start_ADC) > np.array(img_ADC.GetSize())).all(), \
        "Found unrealistically little overlap between T2W and ADC, aborting."
    assert ((idx_end_T2W - idx_start_T2W) > np.array(img_T2W.GetSize())).all(), \
        "Found unrealistically little overlap between T2W and ADC, aborting."
    
    for section in ['features', 'labels']:
        for key in ['img_T2W', 'zonal_mask']:
            slices = [slice(idx_start, idx_end) for (idx_start, idx_end) in zip(idx_start_T2W, idx_end_T2W)]
            if key in sample[section]:
                sample[section][key] = sample[section][key][slices]
    
    for section in ['features', 'labels']:
        for key in ['img_ADC', 'img_DWI', 'lbl']:
            slices = [slice(idx_start, idx_end) for (idx_start, idx_end) in zip(idx_start_ADC, idx_end_ADC)]
            if key in sample[section]:
                sample[section][key] = sample[section][key][slices]
    return sample


def preprocess_scan_dynamic_res(sample, physical_size=(3.6*18, 0.5*144, 0.5*144), center_prostate=False, 
                                center_min_prostate_volume=1, center_max_prostate_volume=None, 
                                align_physical_space=False, resample_uniform_spacing=None, verbose=0):
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
    img_T2W, img_ADC, img_DWI = [sample['features'][key] for key in ['img_T2W', 'img_ADC', 'img_DWI']]
    lbl = sample['labels']['lbl'] if 'lbl' in sample['labels'] else None
    zonal_mask = sample['features']['zonal_mask'] if 'zonal_mask' in sample['features'] else None

    if lbl is not None:
        # check if label has a malignancy to check correct behaviour of this function
        malignant_start = (sitk.GetArrayFromImage(lbl).sum() > 0)

    if align_physical_space:
        # compare physical centers of T2W and ADC scan. The ADC and DWI scans are always the same physical space.
        center_T2W = img_T2W.TransformContinuousIndexToPhysicalPoint(np.array(img_T2W.GetSize())/2.0)
        center_ADC = img_ADC.TransformContinuousIndexToPhysicalPoint(np.array(img_ADC.GetSize())/2.0)

        # calculate distance from center of T2W scan to center of ADC scan
        distance = np.sqrt(np.sum((np.array(center_T2W) - np.array(center_ADC))**2))
        # if difference in center coordinates is more than 2mm, align the scans
        if distance > 2:
            print(f"Aligning scans with distance of {distance:.1f} mm between centers..")
            sample = crop_to_common_physical_space(sample)
    
    # resample scans
    if resample_uniform_spacing is not None:
        # uniform spacing (e.g. for nnUNet)
        res_T2W = resample_uniform_spacing
        res     = resample_uniform_spacing
    else:
        # dynamic spacing
        res_T2W = (0.3, 0.3, 3.6) if 0.3==round(img_T2W.GetSpacing()[0],1) else (0.5, 0.5, 3.6)
        res     = (2.0, 2.0, 3.6)
    img_T2W = resample_img(img_T2W, out_spacing=res_T2W, is_label=False)
    img_ADC, img_DWI = [resample_img(img, out_spacing=res, is_label=False)
                                      for img in (img_ADC, img_DWI)]
    lbl = resample_img(lbl, out_spacing=res, is_label=True) if lbl is not None else None
    zonal_mask = resample_img(zonal_mask, out_spacing=res_T2W, is_label=True) if zonal_mask is not None else None

    # transform images to numpy
    img_T2W, img_ADC, img_DWI, lbl, zonal_mask = [sitk.GetArrayFromImage(img) if img is not None else None
                                                  for img in (img_T2W, img_ADC, img_DWI, lbl, zonal_mask)]
    
    # Center Crop ADC, DWI, Annotation Scans to Same Scope
    zdim       = min(img_T2W.shape[0], img_ADC.shape[0], img_DWI.shape[0])
    xysize     = min(img_T2W.shape[1]*res_T2W[0],img_ADC.shape[1]*res[0],img_DWI.shape[1]*res[0],
                     img_T2W.shape[2]*res_T2W[1],img_ADC.shape[2]*res[1],img_DWI.shape[2]*res[1])
    if lbl is not None:
        zdim   = min(zdim, lbl.shape[0])
        xysize = min(xysize, lbl.shape[1]*res[0], lbl.shape[2]*res[1])
    try:
        xydim_T2W = size_to_dim(xysize, res_T2W[0], assert_multiple=True)
        xydim     = size_to_dim(xysize, res[0],     assert_multiple=True)
    except AssertionError:
        # for an example, see 1215911_96818115
        print(f"Reducing center crop size of {xysize}mm by {xysize} mm to have an integer number of voxels for each scan")
        eps = 1e-12
        assert res_T2W[0]*10 % 1 < eps and res[0] * 10 % 1 < eps, "Failed to convert resolution to integers"
        common_res = compute_lcm(int(res_T2W[0]*10), int(res[0]*10)) / 10
        xysize -= xysize % common_res
        xydim_T2W = size_to_dim(xysize, res_T2W[0], assert_multiple=True)
        xydim     = size_to_dim(xysize, res[0],     assert_multiple=True)
        print(f"New center crop size: {xysize}mm by {xysize} mm.")
    
    img_T2W     = center_crop(img_T2W,    zdim, xydim_T2W, xydim_T2W)
    img_ADC     = center_crop(img_ADC,    zdim, xydim,     xydim)
    img_DWI     = center_crop(img_DWI,    zdim, xydim,     xydim)
    lbl         = center_crop(lbl,        zdim, xydim,     xydim) if lbl is not None else None
    zonal_mask  = center_crop(zonal_mask, zdim, xydim_T2W, xydim_T2W) if zonal_mask is not None else None
    
    # Preprocess and Clean Labels (from Possible Border Interpolation Errors)
    # Deprecated, can have granular annotations: lbl[(lbl!=0)&(lbl!=1)]    = 0 if lbl is not None else None

    # Padding if Z-Dimension is Below Minimum Center-Crop Dimension
    crop_z_dims = size_to_dim(physical_size[0], res[2], assert_multiple=True)
    if ((img_T2W.shape[0] < crop_z_dims) | (img_ADC.shape[0] < crop_z_dims) | (img_DWI.shape[0] < crop_z_dims) | 
        (lbl is not None and lbl.shape[0] < crop_z_dims) | (zonal_mask is not None and zonal_mask.shape[0] < crop_z_dims)):
        img_T2W = resize_image_with_crop_or_pad(img_T2W, img_size=(crop_z_dims,img_T2W.shape[1],img_T2W.shape[2]))
        img_ADC = resize_image_with_crop_or_pad(img_ADC, img_size=(crop_z_dims,img_ADC.shape[1],img_ADC.shape[2]))
        img_DWI = resize_image_with_crop_or_pad(img_DWI, img_size=(crop_z_dims,img_DWI.shape[1],img_DWI.shape[2]))
        lbl     = resize_image_with_crop_or_pad(lbl,     img_size=(crop_z_dims,lbl.shape[1],lbl.shape[2])) if lbl is not None else None
        zonal_mask = resize_image_with_crop_or_pad(zonal_mask, img_size=(crop_z_dims,zonal_mask.shape[1],zonal_mask.shape[2])) if zonal_mask is not None else None
    
    center_coords_T2W, center_coords = None, None
    if center_prostate:
        # calculate number of voxel inside prostate
        nrOfVoxelsInMask = (zonal_mask > 0).sum() # number of voxels in prostate
        
        # calculate volume
        spacing          = res_T2W # spacing is mm/voxel
        voxel_volume     = spacing[0] * spacing[1] * spacing[2] # volume is mm^3 / voxel
        total_volume     = nrOfVoxelsInMask * voxel_volume / 1000 # volume in cc (cm^3)

        # convert a center_max_prostate_volume of -1 to None
        if center_max_prostate_volume == -1:
            center_max_prostate_volume = None

        if total_volume > center_min_prostate_volume and (center_max_prostate_volume is None or total_volume < center_max_prostate_volume):
            bin_mask = (zonal_mask > 0).astype(int)
            # max_area_slice = np.argmax( np.sum(np.sum(bin_mask, axis=-1), axis=-1) )
            # center_coords_T2W  = regionprops(bin_mask[max_area_slice])[0].centroid
            centroid = regionprops(bin_mask)[0].centroid # returns tuple of (z, y, x)
            center_coords_T2W = (centroid[2], centroid[1])
            center_coords = [(v * res_T2W[i] / res[i]) for i, v in enumerate(center_coords_T2W)]

            print(f"Centring scans with centroid of zonal segmentation: {center_coords_T2W} or {center_coords}") if verbose >= 2 else None
        else:
            print("Prostate segmentation not within min/max volume for centring scan") if verbose >= 2 else None

    # Center Crop Volumes to ROI with Same Volume
    crop_xy_dims_T2W = size_to_dim(physical_size[1], res_T2W[1], assert_multiple=True)
    crop_xy_dims     = size_to_dim(physical_size[1], res[1],     assert_multiple=True)
    img_T2W          = center_crop(img_T2W,    crop_z_dims, crop_xy_dims_T2W, crop_xy_dims_T2W, center_2d_coords=center_coords_T2W)
    img_ADC          = center_crop(img_ADC,    crop_z_dims, crop_xy_dims,     crop_xy_dims,     center_2d_coords=center_coords)
    img_DWI          = center_crop(img_DWI,    crop_z_dims, crop_xy_dims,     crop_xy_dims,     center_2d_coords=center_coords)
    lbl              = center_crop(lbl,        crop_z_dims, crop_xy_dims,     crop_xy_dims,     center_2d_coords=center_coords) if lbl is not None else None
    zonal_mask       = center_crop(zonal_mask, crop_z_dims, crop_xy_dims_T2W, crop_xy_dims_T2W, center_2d_coords=center_coords_T2W) if zonal_mask is not None else None
    
    if lbl is not None:
        malignant_end = (lbl.sum() > 0)
        assert malignant_start == malignant_end, "Label has changed due to interpolation/other errors!"

    # pack sample
    sample['features'].update({
        'img_T2W': img_T2W,
        'img_ADC': img_ADC,
        'img_DWI': img_DWI,
    })
    if lbl is not None:
        sample['labels']['lbl'] = lbl
    if zonal_mask is not None:
        sample['features']['zonal_mask'] = zonal_mask

    return sample


def translate_pred_to_reference_scan(pred:np.array, reference_scan:sitk.Image, out_spacing:tuple=(0.5, 0.5, 3.6)) -> sitk.Image:
    """
    EXPERIMENTAL: Translate prediction back to physical space of input T2 scan
    This function performs the reverse operation of the preprocess_study function
    - pred: softmax / binary prediction
    - reference_scan: SimpleITK image to which the prediction should be resampled and resized
    - out_spacing: spacing to which the reference scan is resampled during preprocessing
    """
    reference_scan_resampled = resample_img(reference_scan, out_spacing=out_spacing) # <- should be spacing used to prepare nnUNet data

    # pad softmax prediction to physical size of resampled reference scan (with inverted order of image sizes)
    pred = resize_image_with_crop_or_pad(pred, img_size=list(reference_scan_resampled.GetSize())[::-1])
    pred_itk = sitk.GetImageFromArray(pred)

    # set the physical properties of the predictions
    pred_itk.CopyInformation(reference_scan_resampled)

    # resample predictions to spacing of original reference scan
    pred_itk_resampled = resample_img(pred_itk, out_spacing=reference_scan.GetSpacing())
    return pred_itk_resampled


def read_itk_with_header(path, dtype):
    meta_data = dict()
    itk_img = sitk.ReadImage(path, dtype)

    meta_data['itk_size']      = list(itk_img.GetSize()) #(list-reverse?)
    meta_data['itk_spacing']   = list(itk_img.GetSpacing())
    meta_data['itk_origin']    = list(itk_img.GetOrigin())
    meta_data['itk_direction'] = list(itk_img.GetDirection())
    return itk_img, meta_data


def write_itk_with_header(np_img, path, meta_data):
    itk_img = sitk.GetImageFromArray(np_img)
    itk_img.SetOrigin(meta_data['itk_origin'])
    itk_img.SetSpacing((0.5, 0.5, 3.6))
    itk_img.SetDirection(meta_data['itk_direction']) 

    # write image with failsafe for incomplete write actions
    atomic_image_write(itk_img, path, extension='.nii.gz')



def preprocess_study(subject_id, path_t2w=None, path_adc=None, path_hbv=None, path_seg=None, path_zon=None,
                     newpath_t2w=None, newpath_adc=None, newpath_hbv=None, newpath_seg=None, newpath_zon=None, 
                     physical_size=None, resample_uniform_spacing=None, align_physical_space=True, 
                     apply_binary_smoothing=False, dry_run=False):
    if physical_size is None:
        physical_size = (72.0, 80.0, 80.0)
    if resample_uniform_spacing is None:
        resample_uniform_spacing = (0.5, 0.5, 3.6)

    img_T2W, meta_T2W = read_itk_with_header(path_t2w, sitk.sitkFloat32)
    img_ADC, meta_ADC = read_itk_with_header(path_adc, sitk.sitkFloat32)
    img_DWI, meta_DWI = read_itk_with_header(path_hbv, sitk.sitkFloat32)
    if path_zon is not None: 
        img_ZON, meta_zon = read_itk_with_header(path_zon, sitk.sitkFloat32)
    else:
        img_ZON = None
    if path_seg is not None:
        img_SEG, meta_SEG = read_itk_with_header(path_seg, sitk.sitkUInt8)
    else:
        img_SEG = None

    if img_SEG is not None:
        # convert granular segmentation/AVA to csPCa segmentation
        img_SEG_granular = sitk.GetArrayFromImage(img_SEG)
        mask = (img_SEG_granular == 1) | (img_SEG_granular == 4) | (img_SEG_granular == 5)
        lbl = mask.astype('uint8')
        is_malignant = (lbl.sum() > 0)

        # convert binary label to SimpleITK
        lbl = sitk.GetImageFromArray(lbl)
        lbl.CopyInformation(img_SEG)
    else:
        lbl = None

    # apply preprocessing: resample and centre crop
    sample = {'features': {
        'img_T2W': img_T2W, 'img_ADC': img_ADC, 'img_DWI': img_DWI,
    }, 'labels': {}}
    if img_ZON is not None: sample['features']['zonal_mask'] = img_ZON
    if img_SEG is not None: sample['labels']['lbl'] = lbl
    sample = preprocess_scan_dynamic_res(sample, align_physical_space=align_physical_space, physical_size=physical_size,
                                         resample_uniform_spacing=resample_uniform_spacing)
    img_T2W, img_ADC, img_DWI = [sample['features'][key] for key in ['img_T2W', 'img_ADC', 'img_DWI']]
    if img_SEG is not None: lbl = sample['labels']['lbl']
    if img_ZON is not None: img_ZON = sample['features']['zonal_mask']

    if img_SEG is not None:
        if apply_binary_smoothing:
            # apply binary label smoothing
            for i in range(lbl.shape[0]):
                lbl[i, :, :] = cv2.GaussianBlur(lbl[i, :, :], (9, 9), cv2.BORDER_DEFAULT)

        lbl = (lbl > 0.5).astype('uint8')
        if is_malignant and not lbl.sum() > 0:
            raise AssertionError(f"Label switched value due to preprocessing or binary label smoothing for {subject_id}!")
    
    if dry_run:
        # skip saving images if dry run
        print(f"Skipped saving images & labels for {subject_id} to {newpath_t2w}, ..., {newpath_seg}.")
        return 1
    
    # write images
    write_itk_with_header(img_T2W, newpath_t2w, meta_T2W)
    write_itk_with_header(img_ADC, newpath_adc, meta_T2W)
    write_itk_with_header(img_DWI, newpath_hbv, meta_T2W)
    if img_ZON is not None: write_itk_with_header(img_ZON, newpath_zon, meta_T2W)
    if img_SEG is not None: write_itk_with_header(lbl, newpath_seg, meta_T2W)
    return 1
