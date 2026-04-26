# %%
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

# %%
# Load the breast cancer dataset
cancer = load_breast_cancer(as_frame=True)
X = cancer.data
y = cancer.target
#print(cancer.DESCR)
print(y)

# %%
y_label = [{0: "malignant", 1: "benign"}[i] for i in y]
sns.scatterplot(x=X["mean radius"],
                y=X["mean smoothness"],
                hue=y_label,
                palette="Set1")
# %%
feature_1 = "mean radius"
feature_2 = "mean smoothness"
X = X[[feature_1, feature_2]].values

# %%
# Logistic regression

# BUILD A MODEL: 
model = LogisticRegression(penalty=None).fit(X, y)

# PREDICT AND EVALUATE: 
model.predict_proba(X)
print(model.score(X, y))

# %% Plotting decision boundary

# Create meshgrid
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# Compute decision function over the grid
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.6)  # background
plt.contour(xx, yy, Z, levels=[0], colors='black',
            linewidths=2)  # decision boundary
sns.scatterplot(x=X[:, 0],
                y=X[:, 1],
                hue=y_label,
                edgecolors='k',
                palette="Set1",
                alpha=0.8)
plt.legend()
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title("Logistic Regression Decision Boundary")
plt.show()

# %% DECISION TREE CLASSIFIER
# BUILD A MODEL: 
dt_model = DecisionTreeClassifier(max_depth=3).fit(X, y)
# PREDICT AND EVALUATE: 
print(dt_model.score(X, y))
# %% PLOT DECISION TREE
plot_tree(dt_model, feature_names=[
          feature_1, feature_2], class_names=cancer.target_names, filled=True)
# %% 
# Classification of our data set using the decision tree
#importing necessary libraries
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# %%
# loading the hallmark gene data
hallmarks_data = pd.read_table(
    r"C:\Users\ogeik\OneDrive\Desktop\BME 2315\Module-4-Cancer\data\Menyhart_JPA_CancerHallmarks_core.txt",
    header=None,
    index_col=0
)
# Extract genes associated with "TUMOR-PROMOTING INFLAMMATION"
desired_gene_list = list(
    hallmarks_data.loc["TUMOR-PROMOTING INFLAMMATION"].dropna()
)

# %%
# load expression data and metadata for the validation set
val_data = pd.read_csv(r"C:\Users\ogeik\OneDrive\Desktop\BME 2315\Module-4-Cancer\data\VALIDATION_SET_GSE62944_subsample_log2TPM.csv", index_col=0)
val_meta = pd.read_csv(r"C:\Users\ogeik\OneDrive\Desktop\BME 2315\Module-4-Cancer\data\VALIDATION_SET_GSE62944_metadata.csv", index_col=0)

# Prepare features and labels
def prepare_features(data_df, meta_df, gene_list):
    # Filter for desired genes
    filtered_genes = [gene for gene in gene_list if gene in data_df.index]
    X = data_df.loc[filtered_genes].T  # Transpose to have patients as rows and genes as columns
    y = meta_df.loc[X.index, "cancer_type"] #match each patient in X with its cancer type label from meta_df
    return X, y

X, y = prepare_features(val_data, val_meta, desired_gene_list) # Prepare features and labels

dt_model = DecisionTreeClassifier(max_depth=5) # initialize the decision tree model with a maximum depth of 5
dt_model.fit(X, y) # Fit the model to the data

train_predictions = dt_model.predict(X) # Predict the labels for the training data
print(f"Training Accuracy: {accuracy_score(y, train_predictions):.2f}") # Print the training accuracy

plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=dt_model.classes_, filled=True)
plt.title("Decision Tree: Predicting Cancer Type from Tumor-Promoting Inflammation Genes")
plt.show()
# %%
