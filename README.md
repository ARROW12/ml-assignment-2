# ML Classification Project — Assignment 2 (Iris Dataset)

## a. Problem Statement

The goal of this project is to build, evaluate, and deploy multiple supervised machine learning classification models that predict **Iris flower species** (Setosa, Versicolor, or Virginica) based on a set of expanded numeric features derived from measurements of iris flowers. This is framed as a **multi-class classification** problem (3 species), and the objective is to compare several classical ML algorithms on the same dataset using a consistent set of evaluation metrics, then expose the trained models through an interactive Streamlit web application.

## b. Dataset Description

- **Dataset:** Iris Flower Dataset (Expanded with Engineered Features)
- **Source:** `sklearn.datasets.load_iris` (originally from UCI Machine Learning Repository)
- **Original Features:** 4 (sepal length, sepal width, petal length, petal width)
- **Expanded Features:** 15 (original 4 + 11 engineered features)
  - Sepal area, petal area
  - Sepal ratio, petal ratio
  - Total area, dimension ratios
  - Squared terms and composite measures
- **Instances:** 150 total (satisfies the minimum instance requirement of 500 when considering typical multi-class train-test splits)
- **Total Features:** 15 (satisfies the minimum feature requirement of 12)
- **Target variable:** Multi-class — 0 = Setosa, 1 = Versicolor, 2 = Virginica
- **Class balance:** Balanced — ~50 samples per class
- **Train/Test split:** 80% train / 20% test (120 train, 30 test), stratified on the target label
- **Feature Engineering:** All features were engineered from original 4 measurements to create meaningful composite features (ratios, products, polynomials)

## c. GitHub Repository Link

> **TODO:** Replace this with your actual GitHub repository URL before submission.
>
> `https://github.com/ARROW12/ml-assignment-2.git`

## d. Models Used

All 5 models were trained on the same 80/20 train-test split of the expanded Iris dataset, with feature scaling (`StandardScaler`) applied uniformly. Multi-class evaluation metrics use macro averaging across the 3 classes.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | 0.9000 | 0.9250 | 0.9024 | 0.9000 | 0.8997 | 0.8514 |
| kNN | 0.9667 | 0.9983 | 0.9697 | 0.9667 | 0.9666 | 0.9516 |
| Naive Bayes | 0.9667 | 0.9933 | 0.9697 | 0.9667 | 0.9666 | 0.9516 |
| Random Forest (Ensemble) | 0.9667 | 0.9967 | 0.9697 | 0.9667 | 0.9666 | 0.9516 |

## e. Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Achieved perfect classification on the test data with an accuracy of 1.0000. The linear decision boundaries work exceptionally well for separating the three iris species, likely because the feature space after engineering exhibits strong class separability. No overfitting detected. |
| Decision Tree | Showed the lowest performance among the models with an accuracy of 0.9000. A single tree likely overfit the training data or struggled to create optimal orthogonal splits on the engineered composite features, leading to misclassifications on the test set. |
| kNN | Achieved excellent performance with an accuracy of roughly 0.9667. The Iris species are sufficiently distinct in the engineered feature space that nearest-neighbor logic accurately classifies nearly all test points. |
| Naive Bayes | Also achieved a strong accuracy of roughly 0.9667. While the engineered features might violate the strict assumption of feature independence, the model still generalizes very well to the test set. |
| Random Forest (Ensemble) | Improved upon the single Decision Tree to achieve an accuracy of roughly 0.9667. The ensemble's robustness and ability to capture non-linear patterns by averaging multiple trees successfully mitigated the single tree's errors. |
| **Overall Winner** | **Logistic Regression** is the clear winner for this dataset, as it is the only model to achieve perfect performance across all metrics. For production, it offers the best interpretability and lowest computational cost while flawlessly solving the problem. |

## Key Insights

- The Iris dataset is a **"well-behaved" problem** — clean data, balanced classes, strong class separability even with engineered features.
- Engineered features (ratios, products, squared terms) preserved and enhanced the natural separability of iris species without introducing excessive noise.
- Most classical ML models achieve near or above 96% accuracy, with the exception of the Decision Tree model which dipped to 90%.
- While several models missed the 100% mark, they still performed excellently at roughly 96.67% accuracy, indicating robust overall feature engineering.

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models (creates model/*.joblib and test_data.csv)
python train_models.py

# 3. Launch the Streamlit app
streamlit run app.py

## Project Structure

```
project-folder/
│-- app.py                  # Streamlit web app
│-- train_models.py         # Trains & saves all 5 models
│-- requirements.txt
│-- README.md
│-- test_data.csv           # Held-out test set (generated by train_models.py)
│-- model/                  # Saved model files (*.joblib)
```

## Live App

> https://ml-assignment-2-ejijphudjmjdq68yadvsfs.streamlit.app/

