# Software Vulnerability Prediction in Low-Resource Languages

This is the README file for the reproduction package of the paper: "Software Vulnerability Prediction in Low-Resource Languages: An Empirical Study of CodeBERT and ChatGPT".

## I. Description
This project focuses on predicting software vulnerabilities using GPT-3.5-turbo model. It comprises data preprocessing, model training, and result analysis stages to predict vulnerabilities in software code.

## II. Repository Structure

- **DataPreprocessing.ipynb**: A Jupyter notebook for reading the dataset and removing code comments to prepare the data for model training.
- **OtherLanguageVulPrediction.ipynb**: Implements a GPT model to predict software vulnerabilities. The model is trained using the cleaned dataset to detect and predict potential vulnerabilities.
- **data/**: Contains the original and pre-processed datasets, ready for use in model training and evaluation.
- **utils/**: Holds utility scripts and code snippets used in `DataPreprocessing.ipynb` and `OtherLanguageVulPrediction.ipynb` for data manipulation, preprocessing tasks, and API interactions.

## III. Setup and Usage
### Prerequisites:
- Python (3.x recommended)
- Jupyter Notebook or JupyterLab
- OpenAI API key for using GPT models

### Steps

1. **Install Dependencies**
Install all the required Python packages that list on OtherLanguageVulPrediction.ipynb

2. **Set up your OpenAI API Key**
- Obtain an API key from OpenAI's [developer portal](https://openai.com/).
- In `OtherLanguageVulPrediction.ipynb`, replace `openai.api_key = ""` with your API key, ensuring the key is kept confidential.

### Usage
1. Start Jupyter Notebook or JupyterLab:
jupyter notebook

2. Open DataPreprocessing.ipynb and run the cells to preprocess the dataset.

3. Open OtherLanguageVulPrediction.ipynb to train the model and view the prediction results.