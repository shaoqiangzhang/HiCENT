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

### 2. Converting Data to .npz Format
To convert raw Hi-C data into the .npz format for downstream processing, use the following command:

```python Read_Data.py``` 

 ***Arguments:***

•	-c:  Name of the cell line folder (e.g., GM12878). Default: GM12878. The folder should be located under $root_dir/raw.

•	-hr (optional): Target resolution for the processed data (e.g., 5kb, 10kb). Default: 10kb.

•	-q (optional): Mapping quality of the input data, such as MAPQGE30 or MAPQG0. Default: MAPQGE30.

•	-n (optional): Normalization method, including options like KRnorm or VCnorm. Default: KRnorm.

***Output***

The processed data will be stored in the following directory structure:

```$root_dir/mat/<cell_line_name>```

Each output file will be named based on the chromosome and resolution, following this pattern:```chrN_[HR].npz```

The output format ensures compatibility with downstream tools and simplifies data management.

### 3. Downsampling Data
To perform data downsampling from high-resolution input, execute the following command:

```python Downsample.py```

The following parameters in the ```data_down_parser()``` function within ```Arg_Parser.py``` are appropriately configured under the  following 'Arguments':

•		-c (Cell Line Name): Specifies the cell line to analyze (Default: GM12878).

•		-hr (High-Resolution Files): Specifies the resolution of the high-resolution data (Default: 10kb).

•		-lr (Low-Resolution Files): Specifies the target resolution for downsampled files (Default: 40kb).

•		-r (Downsampling Ratio): Specifies the ratio for downsampling (Default: 16).

The downsampled files will be saved in the same directory as the input files, with filenames formatted as ```chrN_[LR].npz```, where N is the chromosome number and [LR] represents the target low resolution.

### 4. Splitting Data into Train, Validation, and Test Sets

To generate datasets for training, validation, and testing, use the following command: 

```python Generate.py``` 

Before running the script, ensure the default values in the ```data_divider_parser()``` function, located in ```Arg_Parser.py```, are updated to align with your requirements. 
The resulting dataset files will be saved in ```$root_dir/data``` with names like ```hicent_<parameters>.npz```.

## Single-Cell Hi-C Data Preprocessing Workflow

### Preparing the data

This project utilizes single-cell Hi-C data obtained from the schic-topic-model website. The datasets include samples from the following human cell lines: GM12878, H1Esc, HFF, IMR90, and HAP1.

### ```python top100similar.py```
This script processes single-cell Hi-C data by loading contact matrices from text files. It calculates the cosine similarity between these matrices, enabling the identification of the top 100 most similar cells for each cell based on their contact patterns. The output is saved as top_100_similar_cells_cosine.txt, which lists the most similar cells along with their similarity scores.

### ```python pseudobulk.py```
The script is used to generate a pseudo-bulk dataset from single-cell Hi-C data by loading the similarity information between cells, retrieving the top 100 similar cells for each cell, and merging their contact matrices into a pseudo-bulk matrix.

### ```python data_process.py```
Load single-cell and pseudo-bulk contact matrices from the specified directory, perform data preprocessing, split the processed data into training, validation, and test sets, and save them as compressed .npz files.


