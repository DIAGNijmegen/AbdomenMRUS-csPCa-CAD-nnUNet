#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
from picai_baseline.nnunet.softmax_export import \
    save_softmax_nifti_from_softmax
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import (PreprocessingSettings, Sample,
                                      resample_to_reference_scan)


class MissingSequenceError(Exception):
    """Exception raised when a sequence is missing."""

    def __init__(self, name, folder):
        message = f"Could not find scan for {name} in {folder} (files: {os.listdir(folder)})"
        super().__init__(message)


class MultipleScansSameSequencesError(Exception):
    """Exception raised when multiple scans of the same sequences are provided."""

    def __init__(self, name, folder):
        message = f"Found multiple scans for {name} in {folder} (files: {os.listdir(folder)})"
        super().__init__(message)


def strip_metadata(img: sitk.Image) -> None:
    for key in img.GetMetaDataKeys():
        img.EraseMetaData(key)


def convert_to_original_extent(
    pred: np.ndarray,
    pkl_path: Union[Path, str],
    dst_path: Union[Path, str]
) -> sitk.Image:
    # convert to nnUNet's internal softmax format
    pred = np.array([1-pred, pred])

    # read physical properties of current case
    with open(pkl_path, "rb") as fp:
        properties = pickle.load(fp)

    # let nnUNet resample to original physical space
    save_softmax_nifti_from_softmax(
        segmentation_softmax=pred,
        out_fname=str(dst_path),
        properties_dict=properties,
    )

    # now each voxel in softmax.nii.gz corresponds to the same voxel in the original (T2-weighted) scan
    pred_ensemble = sitk.ReadImage(str(dst_path))

    return pred_ensemble


class csPCaAlgorithm(SegmentationAlgorithm):
    """
    Wrapper to deploy nnU-Net model as a grand-challenge.org algorithm.
    """

    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # input / output paths for algorithm
        self.image_input_dirs = [
            "/input/images/transverse-t2-prostate-mri",
            "/input/images/transverse-adc-prostate-mri",
            "/input/images/transverse-hbv-prostate-mri",
        ]
        self.scan_paths = []
        self.cspca_heatmap_path = Path("/output/images/transverse-cancer-heatmap-prostate-mri/heatmap.mha")

        # input / output paths for nnUNet
        self.nnunet_inp_dir = Path("/opt/algorithm/nnunet/input")
        self.nnunet_out_dir = Path("/opt/algorithm/nnunet/output")
        self.nnunet_results = Path("/opt/algorithm/results")

        # ensure required folders exist
        self.nnunet_inp_dir.mkdir(exist_ok=True, parents=True)
        self.nnunet_out_dir.mkdir(exist_ok=True, parents=True)
        self.cspca_heatmap_path.parent.mkdir(exist_ok=True, parents=True)

        # input validation for multiple inputs
        scan_glob_format = "*.mha"
        for folder in self.image_input_dirs:
            file_paths = list(Path(folder).glob(scan_glob_format))
            if len(file_paths) == 0:
                raise MissingSequenceError(name=folder.split("/")[-1], folder=folder)
            elif len(file_paths) >= 2:
                raise MultipleScansSameSequencesError(name=folder.split("/")[-1], folder=folder)
            else:
                # append scan path to algorithm input paths
                self.scan_paths += [file_paths[0]]

    def preprocess_input(self):
        """Preprocess input images to nnUNet Raw Data Archive format"""
        # set up Sample
        sample = Sample(
            scans=[
                sitk.ReadImage(str(path))
                for path in self.scan_paths
            ],
            settings=PreprocessingSettings(
                matrix_size=[20, 160, 160],
                spacing=[3.6, 0.5, 0.5],
            )
        )

        # perform preprocessing
        sample.preprocess()

        # write preprocessed scans to nnUNet input directory
        for i, scan in enumerate(sample.scans):
            path = self.nnunet_inp_dir / f"scan_{i:04d}.nii.gz"
            atomic_image_write(scan, path)

    # Note: need to overwrite process because of flexible inputs, which requires custom data loading
    def process(self):
        """
        Load bpMRI scans and generate heatmap for clinically significant prostate cancer
        """
        # perform preprocessing
        self.preprocess_input()

        # perform inference using nnUNet
        pred_ensemble = None
        ensemble_count = 0
        for trainer in [
            "nnUNetTrainerV2_Loss_CE_checkpoints",
            "nnUNetTrainerV2_Loss_CE_checkpoints2",
            "nnUNetTrainerV2_Loss_CE_checkpoints3",
        ]:
            # predict sample
            self.predict(
                task="Task141_csPCa_semisupervised_PICAI_PubPriv_RUMC",
                trainer=trainer,
                checkpoint="model_best",
            )

            # read softmax prediction
            pred_path = str(self.nnunet_out_dir / "scan.npz")
            pred = np.array(np.load(pred_path)['softmax'][1]).astype('float32')
            os.remove(pred_path)
            if pred_ensemble is None:
                pred_ensemble = pred
            else:
                pred_ensemble += pred
            ensemble_count += 1

        # average the accumulated confidence scores
        pred = pred_ensemble / ensemble_count

        # the prediction is currently at the size and location of the nnU-Net preprocessed
        # scan, so we need to convert it to the original extent before we continue
        pred = convert_to_original_extent(
            pred=pred,
            pkl_path=self.nnunet_out_dir / "scan.pkl",
            dst_path=self.nnunet_out_dir / "softmax.nii.gz",
        )

        # convert heatmap to a SimpleITK image and infuse the physical metadata of original T2-weighted scan
        reference_scan_original_path = str(self.scan_paths[0])
        reference_scan_original = sitk.ReadImage(reference_scan_original_path)
        pred = resample_to_reference_scan(pred, reference_scan_original=reference_scan_original)

        # remove metadata to get rid of SimpleITK warning
        strip_metadata(pred)

        # save prediction to output folder
        atomic_image_write(pred, str(self.cspca_heatmap_path))

    def predict(self, task, trainer="nnUNetTrainerV2", network="3d_fullres",
                checkpoint="model_final_checkpoint", folds="0,1,2,3,4", store_probability_maps=True,
                disable_augmentation=False, disable_patch_overlap=False):
        """
        Use trained nnUNet network to generate segmentation masks
        """

        # Set environment variables
        os.environ['RESULTS_FOLDER'] = str(self.nnunet_results)

        # Run prediction script
        cmd = [
            'nnUNet_predict',
            '-t', task,
            '-i', str(self.nnunet_inp_dir),
            '-o', str(self.nnunet_out_dir),
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


if __name__ == "__main__":
    csPCaAlgorithm().process()
