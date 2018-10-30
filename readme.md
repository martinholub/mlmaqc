---
title: Automating maQC Decision
author: Martin Holub
date: Oct, 30 2018
output: html_document
---

# Introduction {#introduction}
The goal is to train a model implementing `predict` method that can be applied on the runtime of maQCNpipe to data extracted from QC plots to return a confidence estimate on the quality each sample. This *'score'* could be e.g. added to the end of the QC report, to its front page or printed to stdout.

The model would be `pickle.load`ed and the `predict` called by a `python` script called from `R`.

# Easy instal
Install [miniconda](https://conda.io/miniconda.html) and run:
`conda env create --name mlqc -f mlqc.yml`

Note that the name of the environment must be kept in sync in `maQCN_pipeline/maQCNpipe/R/get_quality_metrics.R` (where we make prediction using the model in this environment).


# Data
See [Appendix](#appendix) for information on how data was obtained.

## Quality

I have about 250 experiments, 25 thereof have been rechecked by Pavel (~1 afternoon)

# Model Training Pipeline

The model training pipeline is demonstrated in `notebooks/models.ipynb`. The data required to run it are available at `1090` in `~/mh/ml/ml_data.tar.gz`. Please adjust the paths in notebook for your locations.

## Curent State

All what is described in [Introduction](#introduction) is implemented (with printout to stdout) in function `../maQCNpipe/R/get_quality_metrics.R` for Affymetrix Platform. The trained model is available in `estimators` folder.

## TODO:

- Implement for Agilent, Illumina
- Test if applicable to other species (the model was trained on human data only)
- Give the model information on experiment of origin
  - QC Decision is usually made on relative relationship of a sample to other samples in an
  experiment (potentially also a batch, but this we would like to avoid). Providing the model the information on what are the other samples in given experiment could bring us closer to matching curators' decisions. One way how to do this (indirectly) would be to compute sample-level metrics that contain information on other samples in the experiment. This metric effectively becomes less specific to the sample (which was initial requirement and should be the case for currently implemented metrics), but may help encoding relative relationships within an experiment.

---

# Appendix {#appendix}

## Available Features
There is different data available for different platforms. Either you use all of it and:

a. have different models for each platform, or
b. have some feature transformation at the very input (e.g. PCA), or
c. you use only data common to all platforms and have single model.

Given the current experience, *a.* appears the most suitable option.

### Features List

Common:
- boxplot raw
- raw KDE
- boxplot norm
- norm KDE
- rle_boxplot raw
- rle_boxplot norm
- hierclust
- correlation raw
- correlation norm

Agilent only:
- boxplot negctrl
- boxplot CV
Affy only:
- nuse raw
- nuse norm
- rna_deg
- borderelements

## Getting the data
1. copy experiments from 72/1050 or mount as drive from 1090

```bash
scp -r -3 -i ~/.ssh/id_rsa admin@72-vrt.genevestigator.com:/biodata/ma-raw/{"$(cut -f1 exps_to_process_final.txt | head -n2 | tr '\n' ',' | sed 's/,$//')"} admin@1090-htz.genevestigator.com:/biodata/ma-raw/
```
or better:
```bash
scp exps_to_process_final.txt admin@1090-htz.genevestigator.com:~
scp -r admin@72-vrt.genevestigator.com:/biodata/ma-raw/{"$(cut -f1 exps_to_process_final.txt | tr '\n' ',' | sed 's/,$//')"} /biodata/ma-raw/
# scp -r admin@72-vrt.genevestigator.com:/biodata/ma-raw/{"$(cut -f2 -d" " paritaly_bad_exps_to_process_final_submitqcn.txt | tr '\n' ',' | sed 's/,$//')"} /biodata/ma-raw/
```

or mount the drive, which would not require authentificaiton.

2. use the `<project> <experiment> <platform-type>` format to run the reprocessing with `ML` hardcoded
```bash
~/QC_Normalization_V4/tests/reprocess_exps.sh paritaly_bad_exps_to_process_final_submitqcn.txt
```

3. Pull the information down with
```bash
scp -r admin@1090-htz.genevestigator.com:/biodata/results/NEB_P1/HS/{"$(cut -f2 -d" " labels/exps_to_process_final_submitqcn.txt | tr '\n' ',' | sed 's/,$//')"}/ML/* . > scplog.log
```

Note that some experiments are missing from 72vrt:
HS-01471, HS-01937, HS-00852, HS-02341, HS-01029, HS-01418, HS-02410

## Getting the labels

See `notebooks/labels.ipynb` for approach adopted to obtaining the labels.
