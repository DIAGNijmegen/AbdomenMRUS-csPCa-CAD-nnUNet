FROM python:3.9-slim



RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

# Copy nnU-Net results folder
COPY --chown=algorithm:algorithm nnunet/ /opt/algorithm/nnunet/

# Install algorithm requirements
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

# Copy the processor to the algorithm container folder
COPY --chown=algorithm:algorithm process.py /opt/algorithm/

# Copy your own dependencies to the algorithm container folder
COPY --chown=algorithm:algorithm data_utils.py /opt/algorithm/
COPY --chown=algorithm:algorithm preprocess_data.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=ProstateCancerDetectionContainer
