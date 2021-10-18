import os
import SimpleITK as sitk


'''
Binary PCa Detection in mpMRI
Script:         Preprocessing
Contributor:    joeranbosma
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
'''


def locate_scan(fn, in_dir_scans, nii_fallback=True, generated_fallback=True, manual_fallback=True,
                t2w_snel_fallback=False, inside_patient_dir=False, detection_subdirectories=True,
                verbose=0):
    """
    Locate the file given in 'fn' in one of the folders of the list 'in_dir_scans'. 
    The parameter 'in_dir_scans' can also be a string, in which case only that folder is searched. 
    
    - nii_fallback: whether to also search for '[...].nii' when '[...].nii.gz' is not found.
    - generated_fallback: whether to also search for '[...]_generated.nii.gz' when '[...].nii.gz' is not found.
    - mnt_fallback: whether to also check for 'mnt/projects/[...]'
    - t2w_snel_fallback: whether to also check for '[...]_t2w_snel.mha' when '[...]_t2w.mha' is not found
    - inside_patient_dir: whether to check for scan in patient-subdirectory: '[patient_id]/[...]_[modality].mha'
    - detection_subdirectories: whether to check for scan in subdirectories [from Detection2016, ...]
    """

    if inside_patient_dir:
        # note: fn must be the file itself, so without any leading folders (e.g. '100000_2389497821347162873461234_t2w.mha' or 'M-001_1-2016_sag.mha')
        patient_id = os.path.basename(fn).split("_")[0]
        fn = f"{patient_id}/{fn}"

    if os.path.exists(fn): return fn
    
    # convert in_dir_scans to a list to allow for a list of input directories
    if isinstance(in_dir_scans, str):
        in_dir_scans = [in_dir_scans]
    
    # construct list of allowed file names
    match_filenames = [fn]
    if generated_fallback and '.nii.gz' in fn:
        match_filenames += [fn.replace('.nii.gz', '_generated.nii.gz')]
    if manual_fallback and '.nii.gz' in fn:
        match_filenames += [fn.replace('.nii.gz', '_manual.nii.gz')]
    if nii_fallback and '.nii.gz' in fn:
        match_filenames += [fn.replace('.nii.gz', '.nii') for fn in match_filenames]
    if t2w_snel_fallback and '_t2w.mha' in fn:
        match_filenames += [fn.replace('_t2w.mha', '_t2w_snel.mha')]
    
    # construct list of allowed directories
    match_directories = list(in_dir_scans)
    if detection_subdirectories:
        detection_folders = [
            'Detection2016', 'Detection2017', 'Detection2018', 
            'from Detection2016', 'from Detection2018', 'from Detection2017', 
            'Detection2016_my', 'Detection_new'
        ]
        match_directories += [os.path.join(folder, detection_folder)
                              for folder in match_directories
                              for detection_folder in detection_folders]
    
    if verbose >= 2:
        print("Looking for files named: ")
        print(match_filenames)
        print("In directories:")
        print(match_directories)

    for scan_dir in in_dir_scans:
        for folder in match_directories:
            for fn in match_filenames:
                path = os.path.join(scan_dir, folder, fn)
                if os.path.exists(path):
                    print(f"Found file at {path}") if verbose >= 1 else None
                    return path


def read_scan(subject_id, in_dir_scans, in_dir_annot=None, in_dir_zonal=None, backend='sitk', 
              read_modalities=None, scan_extension="mha", inside_patient_dir=True, t2w_snel_fallback=True,
              assert_scans_are_found=True):
    """
    Read T2w, ADC, DWI and annotation of specified study using either SimpleITK or ANTs
    If in_dir_zonal is specified, also load the prostate segmentation
    """

    if read_modalities is None:
        read_modalities = ['img_T2W', 'img_ADC', 'img_DWI']
    
    # I/O Directories
    lbl_io     = locate_scan(subject_id+'.nii.gz', in_dir_scans=in_dir_annot) if in_dir_annot is not None else None
    
    # Search Directories for MRI Scans
    patient_id = subject_id.split("_")[0]
    if inside_patient_dir:
        fn_base = f"{patient_id}/{subject_id}"
    else:
        fn_base = f"{subject_id}"
    mri_scans = {}
    for modality in read_modalities:
        if modality == 'img_T2W': postfix = 't2w'
        elif modality == 'img_ADC': postfix = 'adc'
        elif modality == 'img_DWI': postfix = 'hbv'
        else: postfix = modality
        img_path = locate_scan(f'{fn_base}_{postfix}.{scan_extension}', in_dir_scans=in_dir_scans, 
                               t2w_snel_fallback=t2w_snel_fallback)
        mri_scans[modality] = img_path

        if assert_scans_are_found:
            assert img_path, f"Not all scans found for subject {subject_id}, {modality} not found!"

    if in_dir_zonal is not None:
        zonal_io = locate_scan(subject_id+'_zones.nii.gz', in_dir_scans=in_dir_zonal)
        if zonal_io is None:
            print(f"Zonal segmentation for {subject_id} not found at {in_dir_zonal}")
    else:
        zonal_io = None

    # Load Data
    zonal_mask = None
    if backend == 'sitk':
        for modality in mri_scans:
            if mri_scans[modality] is not None:
                mri_scans[modality] = sitk.ReadImage(mri_scans[modality], sitk.sitkFloat32)
        
        lbl = sitk.ReadImage(lbl_io) if lbl_io is not None else None

        if zonal_io is not None:
            zonal_mask = sitk.ReadImage(zonal_io)
    else:
        assert False, f"Backend {backend} not recognised. Supported: 'sitk'. "

    # pack images in a dictionary
    sample = {
        'features': mri_scans, 
        'labels': {},
    }

    if lbl is not None:
        sample['labels']['lbl'] = lbl
    
    if in_dir_zonal is not None:
        sample['features']['zonal_mask'] = zonal_mask

    return sample


def atomic_image_write(image, dst_path, extension='.mha', backup_existing_file=False, useCompression=False):
    """Safely write an image to disk, by:
    1. Writing the image to a temporary file: .../[subject_id]_[modality].tmp.mha
    2. IF writing succeeded:
    2a. (optional) rename existing file to .../[subject_id]_[modality].bak.mha
    2b. rename file to target name, which is an atomic operation
    This way, no partially written files should exist at the target path (but could have partial .tmp.mha files)
    """
    assert extension in dst_path, f"Did not find extension ({extension}) in destinateion filename ({dst_path})!"

    # save image to ..tmp.mha
    dst_path_tmp = dst_path.replace(extension, f".tmp{extension}")
    sitk.WriteImage(image, dst_path_tmp, useCompression=useCompression)
    
    # backup existing file
    if backup_existing_file and os.path.exists(dst_path):
        dst_path_bak = dst_path.replace(extension, f".bak{extension}")
        assert not os.path.exists(dst_path_bak), f"There already exists a backup file in {dst_path_bak}!"
        os.rename(dst_path, dst_path_bak)
    
    # rename written file
    os.rename(dst_path_tmp, dst_path)

    return 1
