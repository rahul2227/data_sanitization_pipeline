# Data Sanitization
Project on the data contamination seminar - Goes deeper into data preprocessing pipeline so that LLM's are much more unbiased.

## Data Sanitization Pipeline

### Overview
The **Data Sanitization Pipeline** is a modular software project designed for cleaning training data for language models. It integrates four key modules:
- **Data Preprocessor:** Cleans and normalizes raw text data.
- **Contamination Detector:** Identifies unwanted or benchmark-contaminated data.
- **Membership Inference Checker:** Flags data at risk of memorization.
- **Sanitization Engine:** Aggregates flags and applies cleaning actions (removal, anonymization, rewriting).

![Engine Diagram](./Documentation/diag_and_images/Module%20Diag.png)

### Directory Structure
```tree
data sanitization/
├── data/
├── documentation/
├── exploratory notebooks/
└── src/
    ├── preprocessor/
    │   ├── __init__.py
    │   ├── cleaning.py
    │   ├── deduplication.py
    │   ├── preprocessor_main.py
    │   ├── segmentation.py
    │   └── tokenization.py
    ├── contamination_detector/
    │   ├── __init__.py
    │   ├── detector.py
    │   ├── pacost.py
    │   └── reference_comparison.py
    ├── membership_inference_checker/
    │   ├── __init__.py
    │   ├── embeddings.py
    │   ├── main.py
    │   └── neighborhood.py
    ├── sanitization engine/
    │   └── __init__.py
    └── main.py
```

### Installation
1. Clone the repository
```bash
git clone https://github.com/rahul2227/data-sanitization.git
cd data-sanitization
```

2. Install the packages
```bash
conda env create -f environment.yml 
```

# Individual Commands for each module
## [Data Preprocessor module](./src/preprocessor/README.md)

To preprocess the data you need to run the following command for the execution of the module. 

> **Note:** If you are following the same folder structure as above you just need to run the following command, else you need to enter the arguments for data directories to not have 
> further problems

```bash
 python3 src/preprocessor/preprocessor_main.py --remove-stopwords
```

## [Contamination Detector Module](./src/contamination_detector/README.md)

```bash
python3 src/contamination_detector/detector.py --input-file <input-data-file-path.csv> --output-file <output-data-file-path.csv> --ref_similarity_threshold 0.9 --perplexity_ratio_threshold 0.8
```

for this module you need to specify the input and output files, where the input file will be your preprocessed file and the 
output one will be the contamination flags file

## [Membership Inference Checker Module](./src/membership_inference_checker/README.md)

```bash
python src/membership_inference_checker/main.py --input-file <input-data-file-path.csv> --high-sim-threshold 0.95 --low-sim-threshold 0.3
```

# Sanitization Module

This is the main module for the project, the whole pipeline of the project (including preprocessing, Contamination checker 
and Membership Inference checker can be run from this module as this is a manager module).

**Command to run the whole pipeline**
```bash
python3 src/sanitization_main.py --full-pipeline --use-default-raw-data --sanitization-action anonymize 
```
The above command will run the whole pipeline from scratch, it will use the default wiki dataset as the feed and hence it doesn't need a dataset input parameter by default.
however you can pass other parameters if you deem so necessary.

To know more about other parameters you can look through [Sanitization main](./src/sanitization_main.py)

> **Note:** as this uses subprocesses, the logging is not that robust in a CLI and running full module takes immense amount of time
> (This is especially true for contamination module when run within full sanitization pipeline).
> 
> This warning is there because it freezes the CLI without any output for some time leading to confusion(but have faith I guess that it is running 😅)


> **Disclaimer:** GitHub Copilot is used only for validating pull requests and not for authoring any code. This can be easily verified by seeing commit history.

## Streamlit App (Preview)

An interactive Streamlit UI is provided to run the pipeline step-by-step without the CLI.

- Launch with your conda env (`test_env`):
  - `conda run -n test_env streamlit run streamlit_app.py`
- Upload a CSV with a `text` or `segments` column, or use the demo data.
- Proceed through tabs: Preprocess → Contamination (lite) → Membership → Sanitize.

Note: First run may download models (SentenceTransformers) depending on features used.
