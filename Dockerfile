FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

RUN apt-get -y update
RUN apt-get -y install git

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

# Copy nnU-Net results folder
# The required files for nnUNet inference are:
# results/nnUNet/.../
# |-- plans.pkl
# |-- fold_0/
# |---- model_best.model
# |---- model_best.model.pkl
# |-- fold_1/...
# |-- fold_2/...
# |-- fold_3/...
# |-- fold_4/...
RUN mkdir -p /opt/algorithm/results/ \
    && chown algorithm:algorithm /opt/algorithm/results/
COPY --chown=algorithm:algorithm nnunet/results/ /opt/algorithm/results/

# Install algorithm requirements
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

# Extend the nnUNet installation with custom trainers
COPY --chown=algorithm:algorithm nnUNetTrainerV2_Loss_CE_checkpoints.py /tmp/nnUNetTrainerV2_Loss_CE_checkpoints.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/nnUNetTrainerV2_Loss_CE_checkpoints.py "$SITE_PKG/nnunet/training/network_training/nnUNetTrainerV2_Loss_CE_checkpoints.py"

# Copy the processor to the algorithm container folder
COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=prostate_cancer_detection_processor
