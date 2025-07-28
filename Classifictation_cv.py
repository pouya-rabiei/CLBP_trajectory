# %%
import os
import pandas as pd
import numpy as np
from os.path import join as opj
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap

# Import specific classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier 
from xgboost import XGBClassifier  

# Import for scoring and CV
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    permutation_test_score,
)
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, make_scorer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

# Import for preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)  
# Grouped preprocessing imports
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
import traceback  # For detailed error messages

# Define the path
basepath= '... \\ QBPC\\Traj study'
derivpath = opj(basepath, 'derivatives')
if not os.path.exists(derivpath):
    os.makedirs(derivpath)

# %
# Load the data
df_path = opj(basepath, "data_4cluster.csv")
df = pd.read_csv(df_path)

# %
# Missing data and plot
vars= df.iloc[:, :-1].columns

missed_counts = df.isna().sum()
missed_percent = (missed_counts / df.shape[0]) * 100 

# Calculate range (min-max) and proportion of missing per variable
ranges = df[vars].agg(['min', 'max']).T
ranges['Range'] = ranges['max'] - ranges['min']
ranges = ranges[['min', 'max', 'Range']]

# Creating a DataFrame
df_missed = pd.DataFrame({
    'Variable': missed_counts.index,
    'Missing Count': missed_counts.values,
    'Missing Percent': missed_percent.values,
    'Missing Proportion': missed_counts.values / df.shape[0]
})

df_missed = df_missed.merge(ranges, left_on='Variable', right_index=True, how='left')
df_filtered = df_missed[df_missed['Variable'].isin(vars)].sort_values(by='Missing Count')
df_filtered.to_csv(f'{derivpath}/missing_data_counts.csv', index=False)

# Plotting 
plt.figure(figsize=(8, 6)) 
plt.barh(df_filtered['Variable'], df_filtered['Missing Count'], color='brown')
plt.xlabel('Count of Missing Data', fontsize=11)
plt.ylabel('Variables', fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title('Missing Data Counts', fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{derivpath}/missing_data_counts.svg', bbox_inches='tight')
plt.close()

# %
# Compute the correlation matrix
corr = df.iloc[:, :-1].corr()

# Display the correlation matrix as a heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    vmin=-1,
    vmax=1,
    linewidths=0.3,
    linecolor="white",
    cbar_kws={"label": "Correlation coefficient"},
    square=True,
    annot_kws={"size": 8}
)

plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.title("Correlation Matrix", fontsize=13)
plt.tight_layout()
plt.savefig(f'{derivpath}/Correlation_matrix.svg', bbox_inches='tight')
plt.close()

# Drop stratification column if it exists because colinearity
if "Pain impact stratification" in df.columns:
    df = df.drop(columns=["Pain impact stratification"])
    
# %
# Plot target (cluster) proportion
df['cluster'].value_counts()
plt.figure(figsize=(4, 4))
sns.countplot(x='cluster', data=df, palette='Set2')
plt.xlabel('Cluster', fontsize=11)
plt.ylabel('Count', fontsize=11)
plt.title('Cluster Distribution', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(opj(derivpath, 'cluster_proportion.svg'), bbox_inches='tight')
plt.close()

cluster_counts = df['cluster'].value_counts(normalize=True) * 100  # Get percentages
total_counts = df['cluster'].value_counts() # Get absolute counts
print(cluster_counts)
print(total_counts)

# %
# Extract Features & Target in the dataset
X_df = df.drop(columns=["cluster"])

# Encode target labels to be 0-indexed
label_encoder = LabelEncoder()
y_df = pd.Series(label_encoder.fit_transform(df["cluster"]), name="cluster")
original_class_labels = label_encoder.classes_  # e.g., [1, 2, 3, 4]
num_classes = len(original_class_labels)

print(f"Original class labels mapped: {original_class_labels}")
print(f"Transformed class labels for modeling: {sorted(y_df.unique())}")

# %
# Identifying categorical and numerical features
employment_features = [
    "Full time job", "Part time job", "Unemployed", "Disable for back pain",
    "Disable", "Student","Retired"
]

treatment_features = [
    "Surgery", "Opioids", "Infiltration", "Therapeutic exercise", "Counseling"
]

categorical_features = [
    "Sex", "Kinesiophobia", "Catastrophizing"
] + employment_features + treatment_features

numeric_features = [col for col in X_df.columns if col not in categorical_features]

# %
# Handling missing data
# Define pipeline for categorical and numerical features
# Define a pipeline for categorical features with imputation and one-hot encoding
cat_imputer = Pipeline(
    [
        ("cat_impute", SimpleImputer(strategy="most_frequent")),
        (
            "encode",
            OneHotEncoder(
                drop="if_binary", handle_unknown="ignore", sparse_output=False
            ),
        ),
    ]
)

# Define a pipeline for numerical features with iterative imputation and scaling
num_imputer = Pipeline(
    [("num_impute", IterativeImputer(random_state=42)), ("scale", StandardScaler())]
)
impute_scale = ColumnTransformer(
    [
        ("num", num_imputer, numeric_features),
        ("cat", cat_imputer, categorical_features),
    ],
    verbose_feature_names_out=False,
    remainder="passthrough",
)

# %
# Define classifier models
classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=42, solver="saga"
    ),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Classifier": SVC(probability=True, random_state=42),
    "Multi-Layer Perceptron": MLPClassifier(
        max_iter=2000, random_state=42, early_stopping=True
    ),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    ),
}

# %
# Define hyperparameter grids
param_grids = {
    "Random Forest": {
        'model__n_estimators': [100, 200, 300], 
        'model__max_depth': [3, 5, 10, 20, None],
        'model__min_samples_split': [2, 5, 10], 
        'model__min_samples_leaf': [1, 2, 4]  
    },
    "Logistic Regression": {
        'model__C': [0.01, 0.1, 1, 10], 
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear', 'saga'] 
    },
    "Support Vector Classifier": {
        'model__C': [0.1, 1, 10, 100],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto', 0.01, 0.1] 
    },
    "Multi-Layer Perceptron": {
        'model__hidden_layer_sizes': [(50,), (100,), (50,50)],
        'model__activation': ['relu', 'tanh'],
        'model__alpha': [0.0001, 0.001, 0.01],
        'model__early_stopping': [True],
        'model__learning_rate_init': [0.001, 0.01] #
    },
    "XGBoost": {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.7, 0.8, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 1.0]
    }
}

# Setup CV 
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=123
)  # Can use different random_state for inner

# Initialize dictionaries to store results
nested_results = {}
cluster_f1_scores_dict = {}

# Get indices of categorical features for SMOTENC
categorical_features_indices = [
    X_df.columns.get_loc(col) for col in categorical_features if col in X_df.columns
]

# Function to calculate bootstrap confidence intervals for AUC
def bootstrap_auc_ci(y_true, y_prob, n_bootstraps=1000, alpha=0.95, random_state=None):
    """
    Calculate bootstrap confidence intervals for AUC.
    Parameters:
        y_true: True labels.
        y_prob: Predicted probabilities.
        n_bootstraps: Number of bootstrap samples.
        alpha: Confidence level.
        random_state: Random seed for reproducibility.
    Returns:
        lower: Lower bound of the confidence interval.
        upper: Upper bound of the confidence interval.
    """
    y_true = np.array(y_true).ravel()
    rng = np.random.RandomState(random_state)
    aucs = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        y_true_sample, y_prob_sample = y_true[indices], y_prob[indices]
        if len(np.unique(y_true_sample)) < 2:
            continue
        try:
            aucs.append(
                roc_auc_score(
                    y_true_sample, y_prob_sample, multi_class="ovo", average="macro"
                )
            )
        except ValueError:
            continue
    if not aucs:
        return np.nan, np.nan
    lower, upper = np.percentile(aucs, (1 - alpha) / 2 * 100), np.percentile(
        aucs, (1 + alpha) / 2 * 100
    )
    return lower, upper

# Main modeling loop

# Iterate over each classifier
for name, model_instance in classifiers.items():
    print(f"\n--- Nested CV for {name} ---")

     # Create the pipeline 
     # SMOTENC and the model
    pipeline_with_model = ImbPipeline(
        [
            ("impute_scale", impute_scale),
            (
                "smt",
                SMOTENC(
                    categorical_features=categorical_features_indices, random_state=42
                ),
            ),
            ("model", model_instance),
        ]
    )
    # Create the GridSearchCV object
    grid_search = GridSearchCV(
        estimator=pipeline_with_model,
        param_grid=param_grids[name],
        scoring={"f1_macro": "f1_macro", "roc_auc_ovo": "roc_auc_ovo"},
        refit="f1_macro",  # Refit using the best F1_macro score
        cv=inner_cv,
        n_jobs=-1,
        verbose=0,  # Set verbose=1 or 2 for more GridSearchCV details
    )

    # Initialize lists to store results for this classifier
    outer_fold_y_true, outer_fold_y_pred, outer_fold_y_prob_list = [], [], []
    fold_f1_scores, fold_auc_scores = [], []
    fold_overall_perm_importances_list, fold_class_perm_importances_list = [], []

    all_classes_shap= []
    list_of_shap_values_per_class_fold = [[] for _ in range(num_classes)]
    list_of_X_transformed_test_dfs_fold = []
    fold_feature_names_for_shap = None

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_df, y_df)):
        print(
            f"  Processing Outer Fold {fold_idx + 1}/{outer_cv.get_n_splits()} for {name}..."
        )
        # Split the data into training and testing sets for this fold
        X_train_outer, X_test_outer = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train_outer, y_test_outer = y_df.iloc[train_idx], y_df.iloc[test_idx]

        # Fit the GridSearchCV on the training data of this fold
        grid_search.fit(X_train_outer, y_train_outer)
        # Get the best estimator from the grid search
        best_estimator_for_fold = grid_search.best_estimator_

        # Predict on the test set of this fold
        y_pred_fold = best_estimator_for_fold.predict(X_test_outer)
        y_prob_fold = best_estimator_for_fold.predict_proba(X_test_outer)

        # Store results for this fold
        outer_fold_y_true.extend(y_test_outer.tolist())
        outer_fold_y_pred.extend(y_pred_fold.tolist())
        outer_fold_y_prob_list.append(y_prob_fold)
        
        # Calculate metrics for this fold
        fold_f1_scores.append(f1_score(y_test_outer, y_pred_fold, average="macro"))
        try:
            fold_auc_scores.append(
                roc_auc_score(
                    y_test_outer, y_prob_fold, multi_class="ovo", average="macro"
                )
            )
        except ValueError:
            fold_auc_scores.append(np.nan)

        print(f" Calculating Permutation Importance for fold {fold_idx + 1}...")
        
        # Store fold scores for dot plots
        if "fold_level_results" not in nested_results:
            nested_results["fold_level_results"] = []

        nested_results["fold_level_results"].append({
            "Model": name,
            "Fold": fold_idx + 1,
            "F1 Score": fold_f1_scores[-1],
            "AUC Score": fold_auc_scores[-1] if not np.isnan(fold_auc_scores[-1]) else None
        })
        try:
            # Get the preprocessor and model from the best estimator
            preprocessor_fold = best_estimator_for_fold.named_steps["impute_scale"]
            model_fold = best_estimator_for_fold.named_steps["model"]

            # Transform the test set using the preprocessor
            X_test_outer_transformed_np = preprocessor_fold.transform(X_test_outer)
            # Get the transformed feature names
            transformed_feature_names_fold = preprocessor_fold.get_feature_names_out()
            if fold_feature_names_for_shap is None:
                fold_feature_names_for_shap = transformed_feature_names_fold.tolist()

            # Run permutation importance for the overall model
            perm_res_overall_fold = permutation_importance(
                model_fold,
                X_test_outer_transformed_np,
                y_test_outer,
                scoring="f1_macro",
                n_repeats=200,
                random_state=42 + fold_idx,
                n_jobs=-1,
            )
            # Store overall permutation importances for this fold
            fold_overall_perm_importances_list.append(
                pd.DataFrame(
                    {
                        "feature_name": transformed_feature_names_fold,
                        "feature_importance": perm_res_overall_fold.importances_mean,
                        "fold": fold_idx,
                    }
                )
            )
            # Run permutation importance for each class
            for class_val_idx, original_label_for_naming in enumerate(
                original_class_labels
            ):
                # Transform the label for scoring
                encoded_class_label_for_scoring = label_encoder.transform(
                    [original_label_for_naming]
                )[0]
                # Create a scorer for the specific class
                def f1_for_single_class_fold(y_t_s, y_p_s):
                    return f1_score(
                        y_t_s,
                        y_p_s,
                        labels=[encoded_class_label_for_scoring],
                        average="macro",
                    )
                specific_class_scorer_fold = make_scorer(f1_for_single_class_fold)

                # Calculate permutation importance for the specific class
                perm_res_class_fold = permutation_importance(
                    model_fold,
                    X_test_outer_transformed_np,
                    y_test_outer,
                    scoring=specific_class_scorer_fold,
                    n_repeats=200,
                    random_state=123 + fold_idx + class_val_idx,
                    n_jobs=-1,
                )
                # Store class-specific permutation importances for this fold
                fold_class_perm_importances_list.append(
                    pd.DataFrame(
                        {
                            "feature_name": transformed_feature_names_fold,
                            "feature_importance": perm_res_class_fold.importances_mean,
                            "class_label": original_label_for_naming,
                            "fold": fold_idx,
                        }
                    )
                )
        except Exception as e:
            print(f"    Error PI fold {fold_idx + 1}:")
            traceback.print_exc()

        # IF tree-based model, calculate SHAP values
        print(f"    Calculating SHAP values for fold {fold_idx + 1}...")
        try:

            # Get the explainer for SHAP values
            if name == "Logistic Regression":
                explainer = shap.LinearExplainer(model_fold, masker=X_test_outer_transformed_np)
            elif name in ["Random Forest"]: # ", "XGBoost"
                explainer = shap.TreeExplainer(model_fold)
            # elif name in ["Multi-Layer Perceptron", "Support Vector Classifier"]:
            #     explainer = shap.KernelExplainer(
            #         model_fold.predict_proba,
            #         X_test_outer_transformed_np)
            else:
                raise ValueError(f"SHAP not supported for {name} model.")
            # Transform the test set using the preprocessor
            X_test_outer_transformed_df_fold = pd.DataFrame(
                X_test_outer_transformed_np, columns=transformed_feature_names_fold
            )
            # Calculate SHAP values for the test set of this fold
            shap_values_fold_raw = explainer.shap_values(
                X_test_outer_transformed_df_fold
            )
            # Extract SHAP values for each class
            for class_idx in range(num_classes):
                list_of_shap_values_per_class_fold[class_idx].append(
                    shap_values_fold_raw[:, :, class_idx]
                    
                )
            # Store the transformed test set for SHAP summary plots later
            list_of_X_transformed_test_dfs_fold.append(
                X_test_outer_transformed_df_fold
            )
            
        except Exception as e:
            print(f"    Error SHAP fold {fold_idx + 1}:")
            traceback.print_exc()
            
    # --- End of outer CV fold loop ---

    # Aggregate & Save Permutation Importances  
    if fold_overall_perm_importances_list:
        overall_perm_df_all = pd.concat(
            fold_overall_perm_importances_list, ignore_index=True
        )
        avg_overall_perm_df = (
            overall_perm_df_all.groupby("feature_name")["feature_importance"]
            .agg(["mean", "std"])
            .reset_index()
        )
        avg_overall_perm_df.columns = [
            "feature_name",
            "mean_importance",
            "std_importance",
        ]
        avg_overall_perm_df = avg_overall_perm_df.sort_values(
            by="mean_importance", ascending=False
        )
        avg_overall_perm_df.to_csv(
            opj(derivpath, f"pi_{name}_overall.csv"), index=False
        )
        n_features = len(avg_overall_perm_df["feature_name"].unique())
        fig_height = min(10, 0.4 * n_features + 1.5)  # ~0.4 inches per bar

        plt.figure(figsize=(6, fig_height))
        sns.barplot(
            x="mean_importance",
            y="feature_name",
            data=avg_overall_perm_df,
            color="steelblue"
        )
        plt.xlabel("Mean Permutation Importance", fontsize=11)
        plt.ylabel("Feature", fontsize=11)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title(f"Overall Permutation Importance - {name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(opj(derivpath, f"pi_{name}_overall.svg"), bbox_inches='tight')
        plt.close()

    if fold_class_perm_importances_list:
        class_perm_df_all = pd.concat(
            fold_class_perm_importances_list, ignore_index=True
        )
        avg_class_perm_df = (
            class_perm_df_all.groupby(["feature_name", "class_label"])[
                "feature_importance"
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        avg_class_perm_df.columns = [
            "feature_name",
            "class_label",
            "mean_importance",
            "std_importance",
        ]
        avg_class_perm_df = avg_class_perm_df.sort_values(
            by=["class_label", "mean_importance"], ascending=[True, False]
        )
        avg_class_perm_df.to_csv(
            opj(derivpath, f"pi_{name}_by_class.csv"), index=False
        )
        for class_lbl_plot in avg_class_perm_df["class_label"].unique():
            subset_df = avg_class_perm_df[
                avg_class_perm_df["class_label"] == class_lbl_plot
            ].head(20)  # Top 20 features only

            fig_height = min(10, 0.4 * len(subset_df) + 1.5)
            plt.figure(figsize=(6, fig_height))  # consistent width, adaptive height
            sns.barplot(
                x="mean_importance",
                y="feature_name",
                data=subset_df,
                color="mediumseagreen"
            )
            plt.xlabel("Mean Permutation Importance", fontsize=11)
            plt.ylabel("Feature", fontsize=11)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.title(f"{name} - Class {class_lbl_plot}", fontsize=12)
            plt.tight_layout()
            plt.savefig(opj(derivpath, f"pi_{name}_class_{class_lbl_plot}.svg"), bbox_inches='tight')
            plt.close()

    # Aggregate & Plot SHAP values from CV and compute overall SHAP
    if list_of_X_transformed_test_dfs_fold:
        print(f"  Aggregating SHAP values for {name} from all CV folds...")
        try:
            final_X_test_transformed_all_folds_df = pd.concat(
                list_of_X_transformed_test_dfs_fold, ignore_index=True
            )
            if fold_feature_names_for_shap:
                final_X_test_transformed_all_folds_df.columns = (
                    fold_feature_names_for_shap
                )
            else:
                final_X_test_transformed_all_folds_df.columns = [
                    f"feature_{i}"
                    for i in range(final_X_test_transformed_all_folds_df.shape[1])
                ]
            # Compute Overall SHAP (mean absolute over all classes and samples)
            all_class_shap = []
            for class_idx in range(num_classes):
                if list_of_shap_values_per_class_fold[class_idx]:
                    final_shap_values_for_class = np.concatenate(
                        list_of_shap_values_per_class_fold[class_idx], axis=0
                    )
                    all_class_shap.append(final_shap_values_for_class)
                    if (
                        final_shap_values_for_class.shape[0]
                        == final_X_test_transformed_all_folds_df.shape[0]
                    ):
                        # Create Explanation object for shap plots
                        shap_explanation = shap.Explanation(
                            values=final_shap_values_for_class,
                            data=final_X_test_transformed_all_folds_df.values,
                            feature_names=final_X_test_transformed_all_folds_df.columns.tolist()
                        )

                        # Plot using shap.plots.bar with shap value
                        fig_bar= shap.plots.bar(shap_explanation, max_display= 10)

                        # Plot using shap.plots.bar and dots without shap value
                        plt.figure()
                        class_name_for_plot = original_class_labels[class_idx]
                        shap.summary_plot(
                            final_shap_values_for_class,
                            final_X_test_transformed_all_folds_df,
                            plot_type="bar", max_display= 10,
                            show=False,
                        )
                        plt.xlabel("SHAP Value", fontsize=18
                                   )
                        plt.ylabel("Features", fontsize=18)
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)
                        plt.title(
                            f"SHAP Summary for {name} - Class: {class_name_for_plot}", fontsize=18
                        )
                        plt.tight_layout()
                        plt.savefig(
                            opj(
                                derivpath,
                                f"shap_{name}_class_{class_name_for_plot}.svg",
                            )
                        )
                        plt.close()
                        shap.summary_plot(
                            final_shap_values_for_class,
                            final_X_test_transformed_all_folds_df,
                            plot_type="dot", max_display= 10,
                            show=False,
                        )
                        plt.xlabel("SHAP Value", fontsize=18
                                   )
                        plt.ylabel("Features", fontsize=18)
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)
                        plt.title(
                            f"SHAP Summary for {name} - Class: {class_name_for_plot}", fontsize=18
                        )
                        plt.tight_layout()
                        plt.savefig(
                            opj(
                                derivpath,
                                f"shap_{name}_class_{class_name_for_plot}_dots.svg",
                            )
                        )
                        plt.close()
                    else:
                        print(
                            f"    Dimension mismatch for SHAP aggregation, class {original_class_labels[class_idx]}."
                        )
                        
            if all_class_shap:
                # Stack shape: (num_classes, n_samples, n_features)
                stacked_shap = np.stack(all_class_shap, axis=0)  # shape: (C, S, F)
                abs_shap = np.abs(stacked_shap)
                mean_abs_shap = abs_shap.mean(axis=(0, 1))  # 1D array, shape (F,)
                
                # Create DataFrame of overall SHAP importance
                shap_overall_df = pd.DataFrame({
                    "feature": final_X_test_transformed_all_folds_df.columns,
                    "mean_abs_shap": mean_abs_shap
                }).sort_values(by="mean_abs_shap", ascending=False)

                # Save to CSV
                shap_overall_df.to_csv(
                    opj(derivpath, f"shap_{name}_overall.csv"), index=False
                )
                
                # Create Explanation object for shap plots
                shap_explanation = shap.Explanation(
                values=abs_shap.mean(axis=0),  # Mean absolute SHAP values
                data=np.array(final_X_test_transformed_all_folds_df).astype(float),
                feature_names=final_X_test_transformed_all_folds_df.columns.tolist()
                )
                shap.plots.bar(shap_explanation, max_display=10)
                shap.plots.beeswarm(shap_explanation, max_display=10)
                
                # Plot overall SHAP importance
                plt.figure(figsize=(10, max(6, len(shap_overall_df.head(20)) // 2)))
                sns.barplot(
                    x="mean_abs_shap", 
                    y="feature", 
                    data=shap_overall_df.head(20)
                    )
                plt.xlabel(
                    "Mean |SHAP value| (across classes)", fontsize=18
                    )
                plt.ylabel("Feature", fontsize=18)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.title(
                    f"Overall SHAP Importance for {name}", fontsize=18
                    )
                plt.tight_layout()
                plt.savefig(
                    opj(derivpath, f"shap_{name}_overall_bar.svg")
                    )
                plt.close()
                
        except Exception as e:
            print(f"  Error aggregating SHAP for {name}:")
            traceback.print_exc()

    # --- Permutation Test for the entire pipeline (model + hyperparameter tuning) ---
    print(f"  Performing permutation test for {name} (entire pipeline)...")
    perm_test_orig_score, perm_test_f1_mean, perm_test_f1_std, perm_test_p_value = (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )  # Initialize
    try:
        # skip_perm_test_error # REMOVE TO RUN THE PERMUTATION TEST
        # Re-initialize GridSearchCV for the permutation test to ensure it's fresh.
        grid_search_for_perm_test = GridSearchCV(
            estimator=pipeline_with_model,  # The full pipeline including SMOTENC and model
            param_grid=param_grids[name],
            scoring="f1_macro",
            refit=True,
            cv=inner_cv,  # Inner CV for hyperparameter tuning within permutation_test_score
            n_jobs=-1,
            verbose=0,
        )
        # Perform the permutation test
        score_orig_data, perm_scores, p_value_calc = permutation_test_score(
            grid_search_for_perm_test,
            X_df,
            y_df,
            scoring="f1_macro",
            cv=outer_cv,
            n_permutations=1000,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        perm_test_orig_score = score_orig_data
        perm_test_f1_mean = np.mean(perm_scores)
        perm_test_f1_std = np.std(perm_scores)
        perm_test_p_value = p_value_calc
        print(
            f"    Permutation Test for {name}: Original F1 = {perm_test_orig_score:.4f}, P-value = {perm_test_p_value:.4f}"
        )
    except Exception as e:
        print(f"    Error during permutation test for {name}: {e}")
        traceback.print_exc()
    # # --- End of permutation test ---

    # Aggregate results & metrics
    if outer_fold_y_prob_list:
        y_prob_all_folds = np.concatenate(outer_fold_y_prob_list, axis=0)
    else:
        y_prob_all_folds = (
            np.array([[1 / num_classes] * num_classes] * len(outer_fold_y_true))
            if outer_fold_y_true
            else np.array([])
        )
    mean_f1, std_f1 = (
        (np.mean(fold_f1_scores), np.std(fold_f1_scores))
        if fold_f1_scores
        else (np.nan, np.nan)
    )
    mean_auc, std_auc = (
        (np.nanmean(fold_auc_scores), np.nanstd(fold_auc_scores))
        if fold_auc_scores
        else (np.nan, np.nan)
    )
    auc_ci_l, auc_ci_u = (np.nan, np.nan)
    if (
        outer_fold_y_true
        and y_prob_all_folds.shape[0] == len(outer_fold_y_true)
        and y_prob_all_folds.size > 0
    ):
        auc_ci_l, auc_ci_u = bootstrap_auc_ci(
            outer_fold_y_true, y_prob_all_folds, random_state=42
        )
    if outer_fold_y_true and outer_fold_y_pred:
        cm = confusion_matrix(
            outer_fold_y_true,
            outer_fold_y_pred,
            normalize="true",
            labels=sorted(y_df.unique()),
        )
        plt.figure(figsize=(4, 4))
        sns.heatmap(
            cm,
            annot=True,
            cmap="viridis",
            fmt=".2f",
            xticklabels=original_class_labels,
            yticklabels=original_class_labels,
            cbar_kws={"label": "Proportion"},
            square=True,
            linewidths=0.5,
            linecolor='white'
        )

        plt.title(f"Confusion Matrix - {name}", fontsize=12)
        plt.xlabel("Predicted", fontsize=11)
        plt.ylabel("True", fontsize=11)
        plt.xticks(ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(opj(derivpath, f"cm_{name}.svg"), bbox_inches='tight')
        plt.close()
        
        cluster_f1_scores_dict[name] = f1_score(
            outer_fold_y_true,
            outer_fold_y_pred,
            average=None,
            labels=sorted(y_df.unique()),
        )
    else:
        cluster_f1_scores_dict[name] = [np.nan] * num_classes
    nested_results[name] = {
        "F1 Mean": mean_f1,
        "F1 Std": std_f1,
        "AUC Mean": mean_auc,
        "AUC Std": std_auc,
        "AUC CI Low": auc_ci_l,
        "AUC CI High": auc_ci_u,
        "Permutation Test Orig F1": perm_test_orig_score,  
        "Permutation Test F1 Mean (Permuted)": perm_test_f1_mean,  
        "Permutation Test F1 Std (Permuted)": perm_test_f1_std,  
    }
    
    # Save interim results.
    results_summary_dict = {k: v for k, v in nested_results.items() if k != "fold_level_results"}
    interim_results_df = pd.DataFrame(results_summary_dict).T
    interim_results_df.to_csv(opj(derivpath, "nested_cv_interim_results.csv"))
    # Print
    print(f"  Results for {name}: {nested_results[name]}")

# Save final aggregated results
# Extract and save per-fold results
if "fold_level_results" in nested_results:
    fold_results_df = pd.DataFrame(nested_results["fold_level_results"])
    fold_results_df.to_csv(opj(derivpath, "nested_cv_fold_scores.csv"), index=False)
    
    # Reshape for combined dot plot
fold_results_long = pd.melt(
    fold_results_df,
    id_vars=["Model", "Fold"],
    value_vars=["F1 Score", "AUC Score"],
    var_name="Metric",
    value_name="Score"
)

# Clean up metric names
fold_results_long["Metric"] = fold_results_long["Metric"].str.replace(" Score", "", regex=False)

palette = {
    "AUC": "tab:blue",
    "F1": "tab:red"
}

# Plot both AUC and F1 in one dot plot with hue
plt.figure(figsize=(6, 4))
sns.stripplot(data=fold_results_long, x="Model", y="Score", hue="Metric",
              palette={"AUC": "tab:blue", "F1": "tab:red"}, dodge=True, jitter=0.25, size=6)
plt.title("Per-Fold Scores by Model (AUC & F1)", fontsize=12)
plt.xlabel("Model", fontsize=11)
plt.ylabel("Score", fontsize=11)
plt.xticks(rotation=90, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title="Metric", fontsize=9, title_fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(opj(derivpath, "nested_auc_f1_combined_dotplot.svg"), bbox_inches="tight")
plt.close()
    
results_df = (
    pd.DataFrame({k: v for k, v in nested_results.items() if k != "fold_level_results"})
    .T.sort_values(by="AUC Mean", ascending=False)
)
print("\n--- Overall Nested CV Results ---")
print(results_df)
results_df.to_csv(opj(derivpath, "nested_cv_results_summary.csv"))
if cluster_f1_scores_dict:
    cluster_f1_df = pd.DataFrame(
        cluster_f1_scores_dict, index=original_class_labels
    ).T.sort_values(
        by=original_class_labels[0] if original_class_labels.size > 0 else [],
        ascending=False,
    )
    print("\n--- Per-Cluster F1 Macro (CV) ---")
    print(cluster_f1_df)
cluster_f1_df.plot(kind="bar", figsize=(6, 4), colormap="viridis")
plt.ylabel("F1 Score", fontsize=11)
plt.xlabel("Model", fontsize=11)
plt.title("Per-Cluster F1 Scores", fontsize=12)
plt.xticks(rotation=90, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, title_fontsize=10)
plt.tight_layout()
plt.savefig(opj(derivpath, "nested_f1_per_cluster.svg"), bbox_inches="tight")
plt.close()
    
metrics_to_plot = ["AUC Mean", "F1 Mean"]
plot_df = results_df[metrics_to_plot].copy().dropna()
if not plot_df.empty:
    fig, ax1 = plt.subplots(figsize=(6, 4))
    color1 = "tab:blue"
    ax1.set_xlabel("Model", fontsize=11)
    ax1.set_ylabel("Mean AUC", color=color1, fontsize=11)
    plot_df["AUC Mean"].plot(kind="bar", ax=ax1, color=color1, position=0, width=0.2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticklabels(plot_df.index, fontsize=10)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Mean F1", color=color2, fontsize=11)
    plot_df["F1 Mean"].plot(kind="bar", ax=ax2, color=color2, position=1, width=0.2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Model Comparison: Mean AUC and F1", fontsize=12)
    fig.tight_layout()
    plt.savefig(opj(derivpath, "nested_auc_f1_comparison.svg"), bbox_inches="tight")
    plt.close()

print(f"\nDONE!! All results and plots saved to: {derivpath}")

# %%
