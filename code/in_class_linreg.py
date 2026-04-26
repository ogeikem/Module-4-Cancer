# # %%
# from sklearn.datasets import fetch_california_housing
# from sklearn.linear_model import LinearRegression
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # %%
# housing = fetch_california_housing(as_frame=True)
# print(housing.data.shape, housing.target.shape)
# print(housing.feature_names[0:6])

# # %%
# print(housing.DESCR)
# # %% single feature
# feature = "MedInc"

# X = housing["data"][feature].values.reshape(-1, 1)
# y = housing.target

# # lr_model = LinearRegression()
# # reg = lr_model.fit(X, y)
# reg = LinearRegression().fit(X, y)
# print("R^2", reg.score(X, y))
# print(reg.coef_, reg.intercept_)


# # %%
# x_test = np.linspace(0, 15, 100).reshape(-1, 1)
# y_test = reg.predict(x_test)
# plt.scatter(X, y)
# plt.plot(x_test, y_test, color="red")
# plt.xlabel(feature)
# plt.ylabel("House Value ($100k)")
# plt.annotate(
#     "R^2 = {:.2f}".format(reg.score(X, y)),
#     xy=(0.5, 0.9),
#     xycoords="axes fraction",
#     fontsize=14,
#     ha="center",
# )
# plt.show()

# # %%
# #############################
# # What's the best R^2 for a single feature?
# ##############################

# # Your code here: you can do a loop over all features, or you can just replace the feature name above on line 16
# # instead of housing.feature_names, you could also use housing.data.columns or housing["data"].columns
# for feature in housing.feature_names:
#     print(feature)
#     X = housing["data"][feature].values.reshape(-1, 1)
#     y = housing.target

#     reg = LinearRegression().fit(X, y)
#     print("R^2", reg.score(X, y))
#     print(reg.coef_, reg.intercept_)

# # %% all features: X is the full dataset of all the features instead of just a single one.
# X = fetch_california_housing().data
# y = fetch_california_housing().target
# reg = LinearRegression().fit(X, y)
# print("R^2", reg.score(X, y))
# print(reg.coef_, reg.intercept_)

# # %%


# my application of the code for the project: Linear regression to map tumor states (Stage I through Stage IV) to numeric values (1 to 4)
# We are hoping to use linear regression to test whether inflammation gene expression levels linearly increase with tumor stage.
# Can evaluate the performace of the regression model bysing the R² score (the coefficent of determination) 
# to analyze the linear relationship between inflammation gene expression and tumor stage, where a value close to 1 indicates a strong positive relationshiop; 
# value closer to 0 indicates no linear relationship; and value closer to -1 indicates negative linear relationship.    

# %%
# LINEAR REGRESSION — Tumor Stage Prediction
# Goal: Use inflammation gene expression (via PCA) to predict tumor stage (1–4)
# Author: ChatGPT (with edits done by the team) 
# Date: 04/25/2026; We utilized the code from the lecture and adapted it to our project by applying linear regression to predict tumor stage based on inflammation gene expression levels, which were reduced to principal components (PCs) using PCA. We evaluated the model's performance using R² and MAE metrics, and visualized the results with a scatter plot of predicted vs actual tumor stages.
# AI was utilized to generate some code structure but more importantly to try and understand how to apply linear regression to our specific problem of predicting tumor stage based on gene expression data. The code was adapted to fit the context of our project, including mapping tumor stages to numeric values, performing PCA for dimensionality reduction, and evaluating the model's performance with appropriate metrics.

from turtle import pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# Loading data, just for testing the code in in_class_linreg.py -- should be fine in Jupyter notebook as long as the data files are in the correct path and already run.

base = r"C:\Users\Kidus\OneDrive\Desktop\Computational BME\Module 04\Module-4-Cancer\data"

# Gene expression data
train_data = pd.read_csv(base + r"\TRAINING_SET_GSE62944_subsample_log2TPM.csv", index_col=0)
val_data   = pd.read_csv(base + r"\VALIDATION_SET_GSE62944_subsample_log2TPM.csv", index_col=0)

# Metadata (labels)
train_metadata = pd.read_csv(base + r"\TRAINING_SET_GSE62944_metadata.csv", index_col=0)
val_metadata   = pd.read_csv(base + r"\VALIDATION_SET_GSE62944_metadata.csv", index_col=0)

# %%
# STEP 1 — Map tumor stage strings → numeric values
# --- We convert categorical stages into numbers so regression can work
# =============================================================================

stage_map = {
    "Stage I": 1, "Stage IA": 1, "Stage IB": 1, "Stage IC": 1,
    "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2, "Stage IIC": 2,
    "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3, "Stage IIIC": 3,
    "Stage IV": 4, "Stage IVA": 4, "Stage IVB": 4, "Stage IVC": 4
}

# Extract target variables (y values)
y_train = train_metadata["ajcc_pathologic_tumor_stage"].map(stage_map)
y_val   = val_metadata["ajcc_pathologic_tumor_stage"].map(stage_map)

# Remove samples where stage is missing
train_mask = y_train.notna()
val_mask   = y_val.notna()

X_train_clean = X_train.loc[train_mask]
y_train_clean = y_train.loc[train_mask]

X_val_clean = X_val.loc[val_mask]
y_val_clean = y_val.loc[val_mask]

print("Training samples used:", X_train_clean.shape[0])
print("Validation samples used:", X_val_clean.shape[0])

# %%
# =============================================================================
# STEP 2 — PCA for dimensionality reduction
# WHY: we have many genes → PCA compresses them into fewer features (PCs)

# PCA finds new features (PCs) that capture most of the variance in the data
# We choose n_components=10 to keep the top 10 PCs, which should capture most of the inflammation-related variation
pca = PCA(n_components=10, random_state=42)

# Fit PCA ONLY on training data
X_train_pca = pca.fit_transform(X_train_clean)

# Apply SAME PCA transformation to validation data
X_val_pca = pca.transform(X_val_clean)

print("Variance explained by 10 PCs:",
      pca.explained_variance_ratio_.sum())

# %%
# STEP 3 — Train linear regression model
# Model learns relationship: tumor stage corresponds to the combination of PCA features

reg_model = LinearRegression()

# Fit model using training data
reg_model.fit(X_train_pca, y_train_clean)

# Predict tumor stage for both datasets
y_train_pred = reg_model.predict(X_train_pca)
y_val_pred   = reg_model.predict(X_val_pca)

# %%
# =============================================================================
# STEP 4 — Evaluate model performance
# =============================================================================

# R²: how well predictions match actual values
train_r2 = r2_score(y_train_clean, y_train_pred)
val_r2   = r2_score(y_val_clean, y_val_pred)

# MAE: average prediction error in stage units
train_mae = mean_absolute_error(y_train_clean, y_train_pred)
val_mae   = mean_absolute_error(y_val_clean, y_val_pred)

print("\n--- REGRESSION RESULTS ---")
print(f"Training R²:     {train_r2:.3f}")
print(f"Validation R²:   {val_r2:.3f}")
print(f"Training MAE:    {train_mae:.3f}")
print(f"Validation MAE:  {val_mae:.3f}")

# %%
# STEP 5 — Visualization
# Plot predicted vs actual tumor stage for validation set

plt.figure(figsize=(6,5))

# Scatter plot of predictions
plt.scatter(
    y_val_clean,
    y_val_pred,
    alpha=0.6
)

# Perfect prediction reference line
plt.plot(
    [1, 4],
    [1, 4],
    "r--",
    label="Perfect prediction"
)

plt.xlabel("Actual tumor stage")
plt.ylabel("Predicted tumor stage")
plt.title(
    f"Validation Results\nR² = {val_r2:.3f}, MAE = {val_mae:.3f}"
)
plt.xticks([1,2,3,4])
plt.legend()
plt.tight_layout()
plt.show()