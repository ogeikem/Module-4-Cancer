# Exploratory data analysis (EDA) on a cancer dataset
# Loading the files and exploring the data with pandas
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# %%
# Load the data
####################################################
# data = pd.read_csv(
#     '../Data/TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)  # can also use larger dataset with more genes
# metadata_df = pd.read_csv(
#     '../Data/TRAINING_SET_GSE62944_metadata.csv', index_col=0, header=0)
# print(data.head())

# # %%
# # Explore the data
# ####################################################
# print(data.shape)
# print(data.info())
# print(data.describe())

# # %%
# # Explore the metadata
# ####################################################
# print(metadata_df.info())
# print(metadata_df.describe())

# # %%
# # Subset the data for a specific cancer type
# ####################################################
# cancer_type = 'BRCA'  # Breast Invasive Carcinoma

# # From metadata, get the rows where "cancer_type" is equal to the specified cancer type
# # Then grab the index of this subset (these are the sample IDs)
# cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
# print(cancer_samples)
# # Subset the main data to include only these samples
# # When you want a subset of columns, you can pass a list of column names to the data frame in []
# BRCA_data = data[cancer_samples]

# # %%
# # Subset by index (genes)
# ####################################################
# desired_gene_list = ['TP53', 'BRCA1', 'BRCA2', 'EGFR', 'MYC']
# gene_list = [gene for gene in desired_gene_list if gene in BRCA_data.index]
# for gene in desired_gene_list:
#     if gene not in gene_list:
#         print(f"Warning: {gene} not found in the dataset.")

# # .loc[] is the method to subset by index labels
# # .iloc[] will subset by index position (integer location) instead
# BRCA_gene_data = BRCA_data.loc[gene_list]
# print(BRCA_gene_data.head())

# # %%
# # Basic statistics on the subsetted data
# ####################################################
# print(BRCA_gene_data.describe())
# print(BRCA_gene_data.var(axis=1))  # Variance of each gene across samples
# # Mean expression of each gene across samples
# print(BRCA_gene_data.mean(axis=1))
# # Median expression of each gene across samples
# print(BRCA_gene_data.median(axis=1))

# # %%
# # Explore categorical variables in metadata
# ####################################################
# # groupby allows you to group on a specific column in the dataset,
# # and then print out summary stats or counts for other columns within those groups
# print(metadata_df.groupby('cancer_type')["gender"].value_counts())

# # Explore average age at diagnosis by cancer type
# metadata_df['age_at_diagnosis'] = pd.to_numeric(
#     metadata_df['age_at_diagnosis'], errors='coerce')
# print(metadata_df.groupby(
#     'cancer_type')["age_at_diagnosis"].mean())
# # %%
# # Merging datasets
# ####################################################
# # Merge the subsetted expression data with metadata for BRCA samples,
# # so rows are samples and columns include gene expression for EGFR and MYC and metadata
# BRCA_metadata = metadata_df.loc[cancer_samples]
# BRCA_merged = BRCA_gene_data.T.merge(
#     BRCA_metadata, left_index=True, right_index=True)
# print(BRCA_merged.head())

# # %%
# # Plotting
# ####################################################
# # Boxplot of EGFR expression in BRCA samples using SEABORN
# # Works really well with pandas dataframes, because most methods allow you to pass in a dataframe directly
# sns.boxplot(data=BRCA_merged, x="gender", y='EGFR')
# plt.title("EGFR Expression by Gender in BRCA Samples")
# plt.show()

# # Boxplot of MYC and EGFR expression in BRCA samples using PANDAS directly
# BRCA_merged[['MYC', 'EGFR']].plot.box()
# plt.title("MYC and EGFR Expression in BRCA Samples")
# plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn"])
# %%

# Filtration of the specific gene set (only "Tumor-promoting Inflamation genes") and PCA analysis code:
# for pca, make sure to pip install pandas matplotlib seaborn scikit-learn
hallmarks_data = pd.read_table(
    #r"C:\Users\ogeik\OneDrive\Desktop\BME 2315\Module-4-Cancer\data\Menyhart_JPA_CancerHallmarks_core.txt" -- This is my partner's respective path for reading the cancer hallmarks data
    r"C:\Users\kidus\OneDrive\Desktop\Computational BME\Module 04\Module-4-Cancer\data\Menyhart_JPA_CancerHallmarks_core.txt",
    header=None,
    index_col=0
)

desired_gene_list = list(
    hallmarks_data.loc["TUMOR-PROMOTING INFLAMMATION"].dropna()
)

# there was an issue present intially: Mistake was to use the validation data set intead of the training data set for the PCA and UMAP analyses, which caused a mismatch between the metadata and the gene expression data. 
# The solution was to switch to the training data set for both the gene expression data and the metadata, ensuring that they are aligned and compatible for analysis. This is a common issue when working with multiple datasets, and it's crucial to ensure that the samples in the gene expression data correspond to the samples in the metadata for accurate analysis and visualization.
# using the same data source for varaibles data and metadata; the file sourced is TRAINING_SET_GSE62944_metadata.csv, and this is done in order to just start "training the model" so that we can later apply it to Validation_Set data and finally Test_Set data. 
data = pd.read_csv(
    #r"C:\Users\ogeik\OneDrive\Desktop\BME 2315\Module-4-Cancer\data\TRAINING_SET_GSE62944_subsample_log2TPM.csv" -- This is my partner's respective path for reading the training set data
    r"C:\Users\Kidus\OneDrive\Desktop\Computational BME\Module 04\Module-4-Cancer\data\TRAINING_SET_GSE62944_subsample_log2TPM.csv",
    index_col=0
)
metadata = pd.read_csv(
    #r"C:\Users\ogeik\OneDrive\Desktop\BME 2315\Module-4-Cancer\data\TRAINING_SET_GSE62944_metadata.csv" -- This is my partner's respective path for reading the training set metadata
    r"C:\Users\Kidus\OneDrive\Desktop\Computational BME\Module 04\Module-4-Cancer\data\TRAINING_SET_GSE62944_metadata.csv",
    index_col=0       
)


filtered_genes = [g for g in desired_gene_list if g in data.index]

gene_data = data.loc[filtered_genes]

X = gene_data.T

X.index = X.index.astype(str).str.strip()

X = X.apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, how="all")


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(
    X_pca,
    index=X.index,
    columns=["PC1", "PC2"]
)
pca_df = pca_df.join(metadata["cancer_type"], how="left")
pca_df = pca_df.join(metadata["ajcc_pathologic_tumor_stage"], how="left")

# --- WINDOW 1: CANCER TYPE ---
plt.figure(figsize=(8,6)) # This creates the first window
sns.scatterplot(
    x=pca_df["PC1"],
    y=pca_df["PC2"],
    hue=pca_df["cancer_type"],
    palette="tab20",
    s=60
)
plt.title("PCA: Inflammation by Cancer Type")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(
    title="Cancer Type",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8
)
plt.tight_layout()
plt.show() # Forces window 1 to display and clear the "canvas"

# --- WINDOW 2: TUMOR STAGE ---
plt.figure(figsize=(8,6)) # This creates a COMPLETELY NEW window
sns.scatterplot(
    x=pca_df["PC1"],
    y=pca_df["PC2"],
    hue=pca_df["ajcc_pathologic_tumor_stage"],
    palette="tab20",
    s=60
)
plt.title("PCA: Inflammation by Tumor Stage")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(
    title="Tumor Stage",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8
)
plt.tight_layout()
plt.show() # Forces window 2 to display


# %%

# UMAP analysis code (based on the fitration already done by intial subsetted data and PCA analysis: -- Analysis on cancer type and tumor stage
import umap # Make sure to run 'pip install umap-learn' in your terminal first!

# 1. Instantiate the UMAP model
# n_neighbors: balances local vs global structure (15 is a standard start)
# min_dist: controls how tightly points are packed together (0.1 is standard)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

# 2. Fit and transform the SAME data X that you used for PCA
X_umap = reducer.fit_transform(X)

# 3. Create a DataFrame for UMAP results
umap_df = pd.DataFrame(
    X_umap, 
    index=X.index, 
    columns=["UMAP1", "UMAP2"]
)

# 4. Join both cancer type and tumor stage metadata columns
umap_df = umap_df.join(metadata["cancer_type"], how="left")
umap_df = umap_df.join(metadata["ajcc_pathologic_tumor_stage"], how="left")

# 5. Visualize UMAP colored by cancer type
# Coloring by cancer type allows us to see if the UMAP clusters
# correspond to biologically distinct tumor origins
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=umap_df["UMAP1"],
    y=umap_df["UMAP2"],
    hue=umap_df["cancer_type"],
    palette="tab20",
    s=60
)
plt.title("UMAP: Tumor-Promoting Inflammation Signature and Cancer Type")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(
    title="Cancer Type",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8
)
plt.tight_layout()
plt.show()

# 6. Visualize UMAP colored by tumor stage
# Coloring by tumor stage on the same UMAP embedding allows us to check
# whether the clusters identified above also align with disease progression
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=umap_df["UMAP1"],
    y=umap_df["UMAP2"],
    hue=umap_df["ajcc_pathologic_tumor_stage"],
    palette="tab20",
    s=60
)
plt.title("UMAP: Tumor-Promoting Inflammation Signature and Tumor Stage")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(
    title="Tumor Stage",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8
)
plt.tight_layout()
plt.show()
# %%


# %%
# SETUP FOR REGRESSION
# We already have:
#   X  = training gene expression (from PCA code previously) --- This will all be linked within Juypter Notebook
#   metadata = training metadata (from PCA code previously) --- all again lined up within the Jupyter Notebook
# We will now need to define: 
#   X_train, train_metadata, X_val, val_metadata -- This is for the training and validation datasets, which we will use for training and evaluating our regression model. 
# We will load the validation data and metadata, align them with the training data, and then proceed with the regression analysis.

# Rename training data (from your PCA section)
X_train = X
train_metadata = metadata

# Align training metadata
train_metadata.index = train_metadata.index.astype(str).str.strip()
X_train.index = X_train.index.astype(str).str.strip()
train_metadata = train_metadata.reindex(X_train.index)

# %%
# Load VALIDATION expression + metadata

val_data = pd.read_csv(
    r"C:\Users\Kidus\OneDrive\Desktop\Computational BME\Module 04\Module-4-Cancer\data\VALIDATION_SET_GSE62944_subsample_log2TPM.csv",
    index_col=0
)

val_metadata = pd.read_csv(
    r"C:\Users\Kidus\OneDrive\Desktop\Computational BME\Module 04\Module-4-Cancer\data\VALIDATION_SET_GSE62944_metadata.csv",
    index_col=0
)

# Clean indices
val_data.index = val_data.index.astype(str).str.strip()
val_data.columns = val_data.columns.astype(str).str.strip()
val_metadata.index = val_metadata.index.astype(str).str.strip()

# Build X_val using SAME genes as training
X_val = val_data.loc[filtered_genes].T
X_val.index = X_val.index.astype(str).str.strip()

# Convert to numeric + align columns
X_val = X_val.apply(pd.to_numeric, errors="coerce")
X_val = X_val[X_train.columns]

# Fill missing values using training mean
X_val = X_val.fillna(X_train.mean())

# Align validation metadata
val_metadata = val_metadata.reindex(X_val.index)


# Regression: 

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

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