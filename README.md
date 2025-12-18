# Product Recommendation System

## 1. Description
This project implements a product recommendation system using:
- Apriori algorithm for association rule mining
- MiniBatch KMeans for clustering products by price

---

## 2. Requirements
- Python 3.9 or later
- Windows 64-bit
- Required libraries are provided in the `envi/` folder

---

## 3. Installation

 Install libraries from local folder (offline)
```bash
pip install --no-index --find-links=envi -r requirement.txt
```
---

## 5. Data Preparation

- Place the Amazon Sales Report dataset into the dataset/ directory.
- Run the preprocessing script to clean and transform the raw data:
```bash
python preprocessing.py
```
This step will generate the file new_data_to_analysis, which is required for further analysis.


- Navigate to the test/ directory and generate test cases:
```bash
python create_test_file.py
```
Only after completing these steps should you proceed to run the main system and algorithm tests.
## 4. Running the project

### Run the user interface:
```bash
streamlit run main.py
```
### Test Apriori algorithm:
```bash
python apriori/apriori_test.py
```
### Test Clustering algorithm:
```bash
python clustering/clustering_test.py
```
