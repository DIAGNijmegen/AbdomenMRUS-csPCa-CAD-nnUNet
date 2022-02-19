import os
import SimpleITK
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

# imports required for running nnUNet algorithm
import subprocess
from pathlib import Path

# imports required for my (Joeran) algorithm
import SimpleITK as sitk
from data_utils import atomic_image_write
from preprocessing import preprocess_mpMRI_study, translate_pred_to_reference_scan



class Prostatecancerdetectioncontainer(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # input / output paths for nnUNet
        self.nnunet_input_dir  = Path("/opt/algorithm/nnunet/input")
        self.nnunet_output_dir = Path("/opt/algorithm/nnunet/output")
        self.nnunet_model_dir  = Path("/opt/algorithm/nnunet/results")

        # input / output paths for multiple inputs (bpMRI scan in this case)
        self.t2w_ip_dir        = Path("/input/images/transverse-t2-prostate-mri")
        self.hbv_ip_dir        = Path("/input/images/transverse-hbv-prostate-mri")
        self.adc_ip_dir        = Path("/input/images/transverse-adc-prostate-mri")
        self.output_dir        = Path("/output/images/transverse-cancer-heatmap-prostate-mri")
        self.t2w_image         = Path(self.t2w_ip_dir).glob("*.mha")
        self.hbv_image         = Path(self.hbv_ip_dir).glob("*.mha")
        self.adc_image         = Path(self.adc_ip_dir).glob("*.mha")
        self.heatmap           = self.output_dir / "heatmap.mha"

        # ensure required folders exist
        self.nnunet_input_dir.mkdir(exist_ok=True, parents=True)
        self.nnunet_output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # input validation for multiple inputs
        assert ((len(list(self.t2w_image)) * len(list(self.adc_image)) * len(list(self.hbv_image))) == 1), \
            f"Please upload one .mha file per channel (T2W, ADC, HBV), per job, for grand-challenge.org. Received: " + \
            f"T2W: {list(self.t2w_image)}, ADC: {list(self.adc_image)}, HBV: {list(self.hbv_image)}"

        print(os.listdir(self.t2w_ip_dir))
        print(os.listdir(self.adc_ip_dir))
        print(os.listdir(self.hbv_ip_dir))

        for fn in os.listdir(self.t2w_ip_dir):
            if ".mha" in fn: self.t2w_image = os.path.join(self.t2w_ip_dir, fn)

        for fn in os.listdir(self.adc_ip_dir):
            if ".mha" in fn: self.adc_image = os.path.join(self.adc_ip_dir, fn)

        for fn in os.listdir(self.hbv_ip_dir):
            if ".mha" in fn: self.hbv_image = os.path.join(self.hbv_ip_dir, fn)

    def preprocess_input(self):
        # prepare input images to nnUNet format, with __0000.nii.gz for T2W, __0001.nii.gz for ADC and __0002.nii.gz for HBV
        newpath_t2w = str(self.nnunet_input_dir / "scan_0000.nii.gz")
        newpath_adc = str(self.nnunet_input_dir / "scan_0001.nii.gz")
        newpath_hbv = str(self.nnunet_input_dir / "scan_0002.nii.gz")

        # resample input scans to (0.5, 0.5, 3.6) mm/voxel,
        # or (0.5, 0.5, 3.0) if original through-plane resolution is 3.0 mm/voxel.
        input_scan = sitk.ReadImage(self.t2w_image)
        through_plane_res = 3.6
        if round(input_scan.GetSpacing()[-1], 1) == 3.0:
            through_plane_res = 3.0
        resample_uniform_spacing = (0.5, 0.5, through_plane_res)
        physical_size = (through_plane_res*20, 80.0, 80.0)
        print(f"Resampling scans to {resample_uniform_spacing} mm/voxel, with FOV of {physical_size} mm.")
        all_scan_properties = [
            {
                'input_path': self.t2w_image,
                'output_path': newpath_t2w,
                'type': 'T2W',
            },
            {
                'input_path': self.adc_image,
                'output_path': newpath_adc,
                'type': 'ADC',
            },
            {
                'input_path': self.hbv_image,
                'output_path': newpath_hbv,
                'type': 'HBV',
            },
        ]

        preprocess_mpMRI_study(
            all_scan_properties=all_scan_properties, physical_size=physical_size, subject_id='gc-input',
            resample_uniform_spacing=resample_uniform_spacing, align_physical_space=True,
        )

        return resample_uniform_spacing

    # Note: need to overwrite process because of flexible inputs, which requires custom data loading
    def process(self):
        """
        Load Bi-Parametric MRI (T2W,HBV,ADC) scans and Generate Heatmap for Prostate Cancer  
        """
        # Preprocessing
        resample_uniform_spacing = self.preprocess_input()
        
        # Predict using nnUNet ensemble, averaging multiple restarts
        pred_ensemble = None
        ensemble_count = 0
        for trainer in [
            "nnUNetTrainerV2_Loss_CE_checkpoints",
            "nnUNetTrainerV2_Loss_CE_checkpoints2",
            "nnUNetTrainerV2_Loss_CE_checkpoints3",
        ]:
            self.predict(
                task="Task109_Prostate_mpMRI_csPCa",
                trainer=trainer,
                checkpoint="model_best",
            )
            pred_path = str(self.nnunet_output_dir / "scan.npz")
            pred = np.load(pred_path)['softmax'][1].astype('float32')
            os.remove(pred_path)
            if pred_ensemble is None:
                pred_ensemble = pred
            else:
                pred_ensemble += pred
            ensemble_count += 1

        # Average the accumulated confidence scores
        pred_ensemble /= ensemble_count

        # Convert nnUNet prediction back to physical space of input scan (T2)
        pred_itk_resampled = translate_pred_to_reference_scan_from_file(
            pred = pred_ensemble,
            reference_scan_path = str(self.t2w_image),
            out_spacing = resample_uniform_spacing,
        )

        # save prediction to output folder
        atomic_image_write(pred_itk_resampled, str(self.heatmap), useCompression=True)

        subprocess.check_call(["ls", str(self.output_dir), "-al"])

    def predict(self, task="Task107_Prostate_mpMRI_csPCa", trainer="nnUNetTrainerV2",
                network="3d_fullres", checkpoint="model_best", folds="0,1,2,3,4", 
                store_probability_maps=True, disable_augmentation=False, disable_patch_overlap=False):
        """
        Use trained nnUNet network to generate segmentation masks
        """

        # Set environment variables
        os.environ['RESULTS_FOLDER'] = str(self.nnunet_model_dir)

        # Run prediction script
        cmd = [
            'nnUNet_predict',
            '-t', task,
            '-i', str(self.nnunet_input_dir),
            '-o', str(self.nnunet_output_dir),
            '-m', network,
            '-tr', trainer,
            '--num_threads_preprocessing', '2',
            '--num_threads_nifti_save', '1'
        ]

        if folds:
            cmd.append('-f')
            cmd.extend(folds.split(','))

        if checkpoint:
            cmd.append('-chk')
            cmd.append(checkpoint)

        if store_probability_maps:
            cmd.append('--save_npz')

        if disable_augmentation:
            cmd.append('--disable_tta')

        if disable_patch_overlap:
            cmd.extend(['--step_size', '1'])

        subprocess.check_call(cmd)


def translate_pred_to_reference_scan_from_file(pred=None, pred_path=None, reference_scan_path=None, out_spacing=None):
    """
    Compatibility layer for `translate_pred_to_reference_scan`
    - pred_path: path to softmax / binary prediction
    - reference_scan_path: path to SimpleITK image to which the prediction should be resampled and resized
    - out_spacing: spacing to which the reference scan is resampled during preprocessing

    Returns:
    - SimpleITK Image pred_itk_resampled: 
    """
    if pred is None:
        # read softmax prediction
        pred = np.load(pred_path)['softmax'][1].astype('float32')

    # read reference scan and resample reference to spacing of training data
    reference_scan = sitk.ReadImage(reference_scan_path, sitk.sitkFloat32)

    return translate_pred_to_reference_scan(pred=pred, reference_scan=reference_scan, out_spacing=out_spacing)
    

if __name__ == "__main__":
    Prostatecancerdetectioncontainer().process()
