# hash:sha256:6ceefde681fb5ad2ca41bca5b1ebdfc35ce9ce0f7acc50dc183706cc463f5d61
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-ask-pass /

# Necessary for vtkplotter
RUN apt-get update && apt-get install libgl1 -y
RUN apt-get install libxrender1

RUN pip install -U --no-cache-dir \
    dask[distributed]==2022.11.1 \
    scikit-learn==1.5.1 \
    imlib==0.1.3 \
    git+https://github.com/AllenNeuralDynamics/aind-ng-link.git \
    tqdm==4.65.0 \
    antspyx \
    argschema==3.0.4 \
    ray==2.31.0 \
    pims==0.6.1 \
    xmltodict==0.13.0 \
    boto3==1.26.112 \
    vedo==2023.4.6 \
    notebook \
    point-cloud-utils \
    aind-data-schema==2.8.0 \
    aind-data-access-api==1.10.0
#neuroglancer==2.36 \

# Realign pyopenssl with the cryptography version pulled in above
RUN pip install "pyopenssl==23.2.0" "cryptography==41.0.7"

RUN conda install -c conda-forge k3d pandas && \
    pip install --no-cache-dir numpy==1.26.0

#force rebuild
