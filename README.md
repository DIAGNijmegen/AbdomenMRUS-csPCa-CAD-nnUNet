# Clinically Significant Prostate Cancer Detection in bpMRI using models trained with Report Guided Annotations

## 📖 Citation
If you use this algorithm in your work, please cite our work:

J. S. Bosma, A. Saha, M. Hosseinzadeh, H. Huisman (2021), "_Report Guided Automatic Lesion Annotation for Deep Learning Prostate Cancer Detection in bpMRI_", to be submitted

## Algorithm
This algorithm is hosted on [Grand-Challenge.com](https://grand-challenge.org/algorithms/bpmri-cspca-detection-report-guided-annotations/). 


## Summary
This algorithm predicts a heatmap for the likelihood of clinically significant prostate cancer (csPCa) using biparametric MRI (bpMRI). 
The algorithm ensembles fifteen independent models that were trained jointly on manually and report-guided automatically annotated MRI examinations. 
The heatmap is resampled to the same spatial resolution and physical dimensions as the input T2W image for easier visualisation. 

![Algorithm Overview](https://grand-challenge-public-prod.s3.amazonaws.com/social-images/algorithm/5a0fe3e6-dd36-4b5e-8759-b09cd9177c46/Prostate_MRI_csPCa_Detectio_ov3DZk3.png)


# Mechanism
This algorithm is a deep learning-based detection/diagnosis model, which ensembles 15 independent nnU-Net models  (5-fold cross-validation and 3 restarts). To train these models, a total of **7,430** prostate biparametric MRI (bpMRI) scans paired with a **manual or report-guided automatic annotation** ([PI-RADS v2](https://www.sciencedirect.com/science/article/pii/S0302283815008489?via%3Dihub)) were used. For a description of the report-guided automatic annotation procedure, details on the deep learning model and more, see the associated publication: 

- J. S. Bosma, A. Saha, M. Hosseinzadeh, H. Huisman (2021), "_Report Guided Automatic Lesion Annotation for Deep Learning Prostate Cancer Detection in bpMRI_", to be submitted

**Source Code**: [https://github.com/DIAGNijmegen/Report-Guided-Annotation](https://github.com/DIAGNijmegen/Report-Guided-Annotation/)


## Validation and Performance
This algorithm is evaluated on 300 external visits from Ziekenhuisgroep Twente (ZGT), with histopathological ground truth for all patients. Studies are considered positive if they have at least one Gleason grade group ≥ 2 lesion (csPCa). Each model of the ensemble is also evaluated individually and reflects the performance reported in the accompanying paper, Bosma _et. al._ (2021). 

Patient-based diagnostic performance was evaluated using the Receiver Operating Characteristic (ROC), and summarised to the area under the ROC curve (AUROC). 
Lesion-based diagnostic performance was evaluated using Free-Response Receiver Operating Characteristic (FROC), and summarised to the partial area under the FROC curve (pAUC) between 0.01 and 2.50 false positives per case. 

| Metric                                                        | This algorithm | Models individually | Saha _et. al._ (2021)* | Radiologists |
|---------------------------------------------------------------|----------------|---------------------|-----------------------|--------------|
| AUROC                                                         | 90.1%          | 89.8 ± 1.0%         | 84.0%                 | N/A          |
| pAUC                                                          | 2.117          | 2.063 ± 0.036       | 1.892                 | N/A          |
| Specificity at sensitivity of  <br>radiologists (PI-RADS ≥ 4) | 68.6%          | 64.7 ± 7.1%         | 47.5%                 | 77.9%        |
| Number of training cases                                      | 7,430          | 5,941 ± 9           | 1,584                 | N/A          |

![ROC and FROC curves](https://grand-challenge-public-prod.s3.amazonaws.com/i/2021/11/27/b82ee8d6-d049-45d1-b194-734218e7fc0f.png)

*: The [CAD❋ algorithm proposed in Saha _et. al._ (2021)](https://grand-challenge.org/algorithms/prostate-mri-cad-cspca/) was used to evaluate all 300 visits from ZGT.


## Uses and Directions
- **For research use only**. This algorithm is intended to be used only on biparametric prostate MRI examinations of patients with raised PSA levels or clinical suspicion of prostate cancer. This algorithm should not be used in different patient demographics. 

- **Benefits**: Risk stratification for clinically significant prostate cancer using prostate MRI is instrumental to reduce over-treatment and unnecessary biopsies. 

- **Target population**: This algorithm was trained on patients with raised PSA levels or clinical suspicion of prostate cancer, without prior treatment  (e.g. radiotherapy, transurethral resection of the prostate (TURP), transurethral ultrasound ablation (TULSA), cryoablation, etc.), without prior positive biopsies, without artefacts or and with reasonably-well aligned sequences. 

- **MRI scanner**: This algorithm was trained and evaluated exclusively on prostate bpMRI scans derived from Siemens Healthineers (Skyra/Prisma/Trio/Avanto) MRI scanners with surface coils. It does not account for vendor-neutral properties or domain adaptation, and in turn, is not compatible with scans derived using any other MRI scanner or those using endorectal coils.

- **Sequence alignment and position of the prostate**: While the input images (T2W, HBV, ADC) can be of different spatial resolutions, the algorithm assumes that they are co-registered or aligned reasonably well and that the prostate gland is localized within a volume of 460 cm³ from the centre coordinate.

- **General use**: This model is intended to be used by radiologists for predicting clinically significant prostate cancer in biparametric MRI examinations. The model is not a diagnostic for cancer and is not meant to guide or drive clinical care. This model is intended to complement other pieces of patient information in order to determine the appropriate follow-up recommendation.

- **Appropriate decision support**: The model identifies lesion X as at a high risk of being malignant. The referring radiologist reviews the prediction along with other clinical information and decides the appropriate follow-up recommendation for the patient.

- **Before using this model**: Test the model retrospectively and prospectively on a diagnostic cohort that reflects the target population that the model will be used upon to confirm the validity of the model within a local setting. 

- **Safety and efficacy evaluation**: To be determined in a clinical validation study.


## Warnings
- **Risks**: Even if used appropriately, clinicians using this model can misdiagnose cancer. Delays in cancer diagnosis can lead to metastasis and mortality. Patients who are incorrectly treated for cancer can be exposed to risks associated with unnecessary interventions and treatment costs related to follow-ups. 

- **Inappropriate Settings**: This model was not trained on MRI examinations of patients with prior treatment  (e.g. radiotherapy, transurethral resection of the prostate (TURP), transurethral ultrasound ablation (TULSA), cryoablation, etc.), prior positive biopsies, artefacts or misalignment between sequences. Hence it is susceptible to faulty predictions and unintended behaviour when presented with such cases. Do not use the model in the clinic without further evaluation. 

- **Clinical rationale**: The model is not interpretable and does not provide a rationale for high risk scores. Clinical end users are expected to place the model output in context with other clinical information to make the final determination of diagnosis.

- **Inappropriate decision support**: This model may not be accurate outside of the target population. This model is not designed to guide clinical diagnosis and treatment for prostate cancer. 

- **Generalizability**: This model was primarily developed with prostate MRI examinations from Radboud University Medical Centre and the Andros Kliniek. Do not use this model in an external setting without further evaluation.

- **Discontinue use if**: Clinical staff raise concerns about the utility of the model for the intended use case or large, systematic changes occur at the data level that necessitates re-training of the model.
