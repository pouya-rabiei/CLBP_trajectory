# %%
import os
import pandas as pd
import numpy as np
from pathlib import Path
from os.path import join as opj
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import shap

from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from xgboost import XGBClassifier  

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    permutation_test_score,
)
from sklearn.metrics import (
    balanced_accuracy_score, 
    roc_auc_score, 
    confusion_matrix, 
    make_scorer,
    matthews_corrcoef 
)
from sklearn.calibration import CalibrationDisplay
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    label_binarize
)  
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
import traceback

# %%
# Define the path
basepath = Path.cwd()
derivpath = basepath / "path_to_the_file"
if not os.path.exists(derivpath):
    os.makedirs(derivpath)
    
# %%
# Load the data
df_path = opj(basepath, "file")
df = pd.read_csv(df_path)

# %%
# Missing data and plot
vars= df.iloc[:, :-1].columns

missed_counts = df.isna().sum()
missed_percent = (missed_counts / df.shape[0]) * 100 

# Calculate range (min-max) and proportion of missing per variable
vars = df.columns.drop("cluster", errors="ignore")
num_vars = df[vars].select_dtypes(include=[np.number]).columns

ranges = df[num_vars].agg(["min", "max"]).T
ranges["Range"] = ranges["max"] - ranges["min"]
ranges = ranges[["min", "max", "Range"]]    

# Creating a DataFrame
df_missed = pd.DataFrame({
    "Variable": vars,
    "Missing Count": df[vars].isna().sum().values,
    "Missing Percent": (df[vars].isna().sum().values / df.shape[0]) * 100,
    "Missing Proportion": df[vars].isna().sum().values / df.shape[0],
})

df_missed = df_missed.merge(ranges, left_on="Variable", right_index=True, how="left")
df_filtered = df_missed.sort_values(by="Missing Count")

plt.barh(df_filtered['Variable'], df_filtered['Missing Count'], color='brown')
plt.xlabel('Count of Missing Data', fontsize=16)
plt.ylabel('Variables', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Missing Data Counts', fontsize=16)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{derivpath}/missing_data_counts.svg', 
            dpi=900, 
            bbox_inches='tight', 
            transparent=True
            )
plt.close()

# %%
# Correlation matrix (numeric features only)
num_df = df.drop(columns=["cluster"], errors="ignore").select_dtypes(include=[np.number])
if num_df.shape[1] >= 2:
    corr = num_df.corr()
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
        annot_kws={"size": 9}
    )
    plt.xlabel("Features", fontsize=16)
    plt.ylabel("Features", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Correlation Matrix (numeric features only)", fontsize=16)
    plt.tight_layout()
    plt.savefig(opj(derivpath, "Correlation_matrix_numeric.svg"), 
                dpi=900, 
                transparent=True, 
                bbox_inches="tight")
    plt.close()

# %%
# Drop stratification column if it exists because colinearity
if "Pain impact" in df.columns:
    df = df.drop(columns=["Pain impact"])
    
# %%
# Plot cluster proportion
df['cluster'].value_counts()
sns.countplot(x='cluster', data=df, palette='Set2')
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Cluster Distribution', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(opj(derivpath, 'cluster_proportion.svg'), dpi=900, transparent=True,
            bbox_inches='tight')
plt.close()

cluster_counts = df['cluster'].value_counts(normalize=True) * 100  
total_counts = df['cluster'].value_counts() 
print(cluster_counts)
print(total_counts)

# %%
# Extract Features & Target in the dataset
X_df = df.drop(columns=["cluster", "id"])

# Encode target labels to be 0-indexed
label_encoder = LabelEncoder()
y_df = pd.Series(label_encoder.fit_transform(df["cluster"]), name="cluster")
original_class_labels = label_encoder.classes_  
num_classes = len(original_class_labels)

print(f"Original class labels mapped: {original_class_labels}")
print(f"Transformed class labels for modeling: {sorted(y_df.unique())}")

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y_df,
    test_size=0.20,
    stratify=y_df,
    random_state=42,
)

# %%
# Identifying categorical and numerical features
employment_features = [
    "Full time job", "Part time job", "Unemployed", "Disabled", "Student", "Retired"
]

treatment_features = [
    "Surgery", "Opioids", "Infiltration", "Therapeutic exercise", "Counseling"
]

categorical_features = [
    "Sex", "Kinesiophobia", "Catastrophizing"
 ] + employment_features + treatment_features

numeric_features = [col for col in X_df.columns if col not in categorical_features]

# %%
# Define preprocessing pipelines for numerical and categorical data
cat_imputer = Pipeline(
    [
        ("cat_impute", IterativeImputer(estimator=LogisticRegression(max_iter=1000),
                                        random_state=42)),
        (
            "encode",
            OneHotEncoder(
                drop="if_binary", handle_unknown="ignore", sparse_output=False
            ),
        ),
    ]
)

num_imputer = Pipeline(
    [("num_impute", IterativeImputer(random_state=42)), ("scale", StandardScaler())] # default estimator=BayesianRidge() 
)
impute_scale = ColumnTransformer(
    [
        ("num", num_imputer, numeric_features),
        ("cat", cat_imputer, categorical_features),
    ],
    verbose_feature_names_out=False,
    remainder="passthrough",
)

# %%
# Define classifier models
classifiers = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", 
        max_iter=2000, 
        random_state=42, 
        solver="saga"
    ),
    
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", 
        random_state=42
    ),
    
    "Support Vector Classifier": SVC(
        class_weight="balanced", 
        probability=True, 
        random_state=42
    ),
    
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    ),
}

# %%
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
    "XGBoost": {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.7, 0.8, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 1.0]
    }
}

# %%
# Nested Cross-Validation setup
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=123
)  
scoring={
    "AUROC": "roc_auc_ovo",
    "BA": "balanced_accuracy",
    "MCC": make_scorer(matthews_corrcoef),
}

# %%
results = []
fold_rows = []

for name, model_instance in classifiers.items():
    print(f"\n--- Nested CV for {name} ---")

    pipeline_with_model = ImbPipeline([
        ("impute_scale", impute_scale),
        ("model", model_instance),
    ])

    grid_search = GridSearchCV(
        estimator=pipeline_with_model,
        param_grid=param_grids[name],
        scoring=scoring,
        refit="AUROC",
        cv=inner_cv,
        n_jobs=-1,
        verbose=4,
    )

    # Outer CV -> OOF metrics
    oof_true = []
    oof_prob = []

    for fold_idx, (tr_idx, va_idx) in enumerate(outer_cv.split(X_train, y_train), start=1):
        print(f"  Processing Outer Fold {fold_idx}/{outer_cv.get_n_splits()} for {name}...")

        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        grid_search.fit(X_tr, y_tr)

        best_fold = grid_search.best_estimator_
        prob_va = best_fold.predict_proba(X_va)

        oof_true.append(y_va.to_numpy())
        oof_prob.append(prob_va)

        pred_va = np.argmax(prob_va, axis=1)
        auc_fold = roc_auc_score(y_va, prob_va, multi_class="ovo", average="macro")
        ba_fold  = balanced_accuracy_score(y_va, pred_va)
        mcc_fold = matthews_corrcoef(y_va, pred_va)

        fold_rows.append({
            "Model": name,
            "Fold": fold_idx,
            "AUROC": auc_fold,
            "BA": ba_fold,
            "MCC": mcc_fold,
        })

    y_oof_true = np.concatenate(oof_true, axis=0)
    y_oof_prob = np.concatenate(oof_prob, axis=0)
    y_oof_pred = np.argmax(y_oof_prob, axis=1)

    auc_oof = roc_auc_score(y_oof_true, y_oof_prob, multi_class="ovo", average="macro")
    ba_oof  = balanced_accuracy_score(y_oof_true, y_oof_pred)
    mcc_oof = matthews_corrcoef(y_oof_true, y_oof_pred)

    # Final refit on FULL TRAIN
    grid_search.fit(X_train, y_train)

    best_final  = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("  Best params (from full TRAIN grid):", best_params)

    # Evaluate once on TEST
    y_test_prob = best_final.predict_proba(X_test)
    y_test_pred = np.argmax(y_test_prob, axis=1)

    auc_test = roc_auc_score(y_test, y_test_prob, multi_class="ovo", average="macro")
    ba_test  = balanced_accuracy_score(y_test, y_test_pred)
    mcc_test = matthews_corrcoef(y_test, y_test_pred)
    
    # Per class AUC score
    y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))

    per_class_auc = {}
    for c, cls in enumerate(original_class_labels):
        auc_c = roc_auc_score(
            y_test_bin[:, c],
            y_test_prob[:, c]
        )
        per_class_auc[f"AUROC_class_{cls}"] = auc_c
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(num_classes), normalize="true")
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=original_class_labels,
                yticklabels=original_class_labels)
    plt.title(f"Confusion Matrix - {name}", fontsize=16)
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("True", fontsize=16)
    plt.xticks(ha="right", fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(opj(derivpath, f"cm_{name}.svg"), 
                bbox_inches="tight", 
                dpi=900, 
                transparent=True)
    plt.close()
    
    # Calibration plot
    assert y_test_prob.shape[1] == num_classes
    cmap = plt.get_cmap("tab10")
    class_colors = {
        c: cmap(c) for c in range(num_classes)
    }
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2, figure=fig)
    ax_cal = fig.add_subplot(gs[:2, :])
    # Curves (one-vs-rest per class)
    for c, cls in enumerate(original_class_labels):
        y_true_bin = (y_test.to_numpy() == c).astype(int)
        y_prob_c = y_test_prob[:, c]

        CalibrationDisplay.from_predictions(
            y_true_bin,
            y_prob_c,
            n_bins=10,
            name=f"Class {cls}",
            ax=ax_cal,
            ref_line=True,
            color=class_colors[c],
        )
    ax_cal.set_title(f"Calibration curve - {name} (one-vs-rest)", fontsize=16)
    ax_cal.set_xlabel("Mean predicted probability", fontsize=16)
    ax_cal.set_ylabel("Fraction of positives", fontsize=16)
    ax_cal.tick_params(axis="x", labelsize=16)
    ax_cal.tick_params(axis="y", labelsize=16)
    ax_cal.grid(True)

    # Histograms
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    for c, cls in enumerate(original_class_labels):
        r, col = grid_positions[c]
        ax_h = fig.add_subplot(gs[r, col])
        ax_h.hist(
            y_test_prob[:, c], 
            bins=10, 
            range=(0, 1),
            color=class_colors[c]
            )
        ax_h.set_title(f"Class {cls}", fontsize=16)
        ax_h.set_xlabel(f"Predicted probability (class {cls})", fontsize=16)
        ax_h.set_ylabel("Count", fontsize=16)
        ax_h.tick_params(axis="x", labelsize=16)
        ax_h.tick_params(axis="y", labelsize=16)
        ax_h.grid(True)

    plt.tight_layout()
    plt.savefig(opj(derivpath, f"calibration_{name}.svg"),
                bbox_inches="tight", dpi=900, transparent=True)
    plt.close()
    
    # Permutation test 
    fit_params_perm = {}
        
    perm_orig, perm_mean, perm_std, perm_p = np.nan, np.nan, np.nan, np.nan
    try:
        grid_for_perm = GridSearchCV(
            estimator=pipeline_with_model,
            param_grid=param_grids[name],
            scoring=scoring,
            refit="AUROC",
            cv=inner_cv,
            n_jobs=-1,
            verbose=0,
        )
        perm_orig, perm_scores, perm_p = permutation_test_score(
            grid_for_perm,
            X_train,
            y_train,
            scoring=scoring["AUROC"],
            cv=outer_cv,
            n_permutations=1000,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        perm_mean = float(np.mean(perm_scores))
        perm_std  = float(np.std(perm_scores))
        print(f"  Permutation test (TRAIN only): orig={perm_orig:.4f} p={perm_p:.4g}")
    except Exception:
        print(f"  Permutation test failed for {name}")
        traceback.print_exc()

    # SHAP on FULL TRAIN final model (per class + overall)
    try:
        pre = best_final.named_steps["impute_scale"]
        model = best_final.named_steps["model"]

        X_train_trans = pre.transform(X_train)
        feat_names = pre.get_feature_names_out()
        X_train_trans_df = pd.DataFrame(X_train_trans, columns=feat_names)

        if name == "Logistic Regression":
            explainer = shap.LinearExplainer(model, X_train_trans_df)
            shap_vals = explainer.shap_values(X_train_trans_df)
        elif name in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_train_trans_df)
        elif name == "Support Vector Classifier":
            explainer = shap.KernelExplainer(model.predict_proba, X_train_trans_df)
        else:
            shap_vals = None

        if shap_vals is None:
            print(f"  SHAP skipped for {name}")
        else:
            if isinstance(shap_vals, list):
                per_class = [np.asarray(v) for v in shap_vals]  # list of (N,F)
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                per_class = [shap_vals[:, :, c] for c in range(num_classes)]  # (N,F) each
            else:
                raise ValueError(f"Unexpected SHAP output for {name}: {type(shap_vals)}")

            # Per-class
            for c, cls in enumerate(original_class_labels):
                sv = per_class[c]
                imp = np.mean(np.abs(sv), axis=0)

                df_imp = pd.DataFrame({
                    "feature": feat_names,
                    "mean_abs_shap": imp
                }).sort_values("mean_abs_shap", ascending=False)

                df_imp.to_csv(opj(derivpath, f"shap_{name}_class_{cls}.csv"), index=False)

                plt.figure()
                shap.summary_plot(sv, X_train_trans_df, plot_type="dot", max_display=10, show=False)
                plt.title(f"SHAP Importance for - {name} - Class: {cls}", fontsize=16)
                plt.xlabel("SHAP value", fontsize=16)
                plt.ylabel("Feature", fontsize=16)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.tight_layout()
                plt.savefig(opj(derivpath, f"shap_{name}_class_{cls}.svg"),
                            bbox_inches="tight", dpi=900, transparent=True)
                plt.close()

            # Overall across classes
            stacked_shap = np.stack([np.abs(per_class[c]) for c in range(num_classes)], axis=0)  # (C,N,F)
            mean_abs_shap = stacked_shap.mean(axis=(0, 1))

            shap_overall_df = pd.DataFrame({
                "feature": feat_names,
                "mean_abs_shap": mean_abs_shap
            }).sort_values("mean_abs_shap", ascending=False)

            shap_overall_df.to_csv(opj(derivpath, f"shap_{name}_overall.csv"), index=False)
            
            overall_values = np.mean(np.stack([np.abs(v) for v in per_class], axis=0), axis=0)  # (N, F)
            shap_exp = shap.Explanation(
                values=overall_values,
                data=X_train_trans_df.values,
                feature_names=list(feat_names),
            )
            plt.figure()
            shap.plots.beeswarm(shap_exp, max_display=10, show=False)
            plt.title(f"Overall SHAP Importance for {name}", fontsize=16)
            plt.xlabel("SHAP value", fontsize=16)
            plt.ylabel("Feature", fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            plt.savefig(opj(derivpath, f"shap_{name}_overall.svg"),
                        dpi=900, 
                        bbox_inches="tight", 
                        transparent=True
                        )
            plt.close()

    except Exception:
        print(f" SHAP failed for {name}")
        traceback.print_exc()

    results.append({
        "Model": name,
        "Train_OOF_AUROC": auc_oof,
        "Train_OOF_BA": ba_oof,
        "Train_OOF_MCC": mcc_oof,
        "Test_AUROC": auc_test,
        "Test_BA": ba_test,
        "Test_MCC": mcc_test,
        **per_class_auc,
        "Permutation_Orig_AUROC": perm_orig,
        "Permutation_Mean_AUROC": perm_mean,
        "Permutation_Std_AUROC": perm_std,
        "Permutation_p": perm_p,
        "BestParams": str(best_params),
    })
results_df = pd.DataFrame(results)
results_df.to_csv(opj(f"{derivpath}/results_summary.csv"), index=False)
print(results_df[[
    "Model", 
    "Train_OOF_AUROC", 
    "Test_AUROC", 
    "Train_OOF_BA", 
    "Test_BA", 
    "Train_OOF_MCC", 
    "Test_MCC"
    ]])

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(opj(f"{derivpath}/fold_scores_train_only.csv"), index=False)

if not fold_df.empty:
    fold_long = fold_df.melt(
        id_vars=["Model", "Fold"],
        value_vars=["AUROC", "BA", "MCC"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(6, 4))
    sns.stripplot(data=fold_long, x="Model", y="Score", hue="Metric",
                  dodge=True, jitter=0.25)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel("Score", fontsize=16)
    plt.title("Nested CV fold scores", fontsize=16)
    plt.tight_layout()
    plt.savefig(opj(derivpath, "train_nested_fold_scores_dotplot.svg"),
                bbox_inches="tight", dpi=900, transparent=True)
    plt.close()
    
class_auc_df = results_df.melt(
    id_vars="Model",
    value_vars=["AUROC_class_1", "AUROC_class_2", "AUROC_class_3", "AUROC_class_4"],
    var_name="Cluster",
    value_name="AUROC"
)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=class_auc_df,
    x="Model",
    y="AUROC",
    hue="Cluster",
    errorbar=None
)
plt.xticks(rotation=45, ha="right", fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Classifier", fontsize=16)
plt.ylabel("AUROC", fontsize=16)
plt.ylim(0.5, 1.0)
plt.title("Per-class AUROC by classifier", fontsize=16)
plt.tight_layout()
plt.savefig(opj(derivpath, "Per-class AUROC by classifier.svg"),
                bbox_inches="tight", dpi=900, transparent=True)
plt.show()
    
print(f"\nDONE. Outputs saved in: {derivpath}")
