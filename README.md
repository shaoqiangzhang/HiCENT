# HiCENT
Enhancing sngle-cell and bulk Hi-C data by generative Transformer model


## Dependencies
•	Python 3.6

•	numpy: 1.19.5

•	pandas: 1.1.5

•	matplotlib: 3.3.4

•	scikit-learn: 0.24.2

•	torch: 1.10.0+cu111

## Bulk Hi-C Data Preprocessing Workflow

This project utilizes Hi-C data from Rao et al., 2014 (GEO Accession: GSE63525). https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525 Specifically, the intrachromosomal contact matrices from GM12878, K562, and CH12-LX (mouse) cell lines are processed through the following stages:

### 1. Preparing the Data
To prepare the Hi-C datasets for analysis, follow these steps:

***1.1 Define the Root Directory***

Specify the root directory for the project by setting the root_dir variable in Datasets /Arg_Parser.py as a string. For example:
```root_dir = './Datasets'```

***1.2 Create a Subdirectory for Raw Data***

Within the defined root directory, create a subdirectory named raw to store the unprocessed datasets:
```mkdir $root_dir/raw```

***1.3 Download and Extract Hi-C Data***

Download the required Hi-C datasets and extract them into the raw directory. Organize the extracted files into subdirectories named after the respective cell lines (e.g., GM12878, K562, CH12-LX). Each subdirectory should contain the contact matrices for various chromosomes and resolutions.
