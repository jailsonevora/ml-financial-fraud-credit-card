import gc
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from anti_money_laundering.config import FIGURES_DIR, PROCESSED_DATA_DIR
import matplotlib.pyplot as plt
import warnings
import numpy as np    
import cirq  
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from cirq import Simulator
import qutip as qt
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import plotly.express as px

def set_defaults():
    # Set Matplotlib defaults
    #plt.style.use("seaborn-whitegrid")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rc("figure", autolayout=True)
    plt.rc(
        "axes",
        labelweight="bold",
        labelsize="large",
        titleweight="bold",
        titlesize=14,
        titlepad=10,
    )
    # Mute warnings
    warnings.filterwarnings('ignore')

def visualize_quantum_data(circuits_X_train):
    # Your circuits
    circuits_X_train = circuits_X_train
    
    # Extract Rx angles for qubit (0, 0)
    angles = []
    target_qubit = cirq.GridQubit(0, 0)
    
    for circuit in circuits_X_train:
        for moment in circuit:
            for op in moment.operations:
                if target_qubit in op.qubits and type(op.gate).__name__ == "Rx":
                    angles.append(op.gate.exponent * np.pi)  # Corrected here
    
    # Initialize Bloch sphere
    bloch = qt.Bloch()
    qubit_state = qt.basis(2, 0)  # |0⟩
    
    # Apply Rx rotations and add states to Bloch sphere
    for angle in angles:
        Rx_gate = np.cos(angle / 2) * qt.qeye(2) - 1j * np.sin(angle / 2) * qt.sigmax()
        qubit_state = Rx_gate * qubit_state
        bloch.add_states(qubit_state)    
    bloch.show()
    
def visualize_single_qubit_bloch_cirq(circuits, qubit_index=0, max_display=5):
    """
    Visualizes the Bloch vector of a specific qubit in a quantum circuit.

    Args:
        circuits (list of cirq.Circuit): Quantum circuits to visualize.
        qubit_index (int): The qubit index to visualize.
        max_display (int): Max number of circuits to visualize.

    Returns:
        None (shows Bloch sphere plots).
    """
    simulator = Simulator()  # Using the correct Simulator class

    # Iterate through the circuits and visualize the state of the specified qubit
    for i, circuit in enumerate(circuits[:max_display]):
        # Simulate the circuit to get the final state vector
        result = simulator.simulate(circuit)
        state_vector = result.final_state_vector

        # Extract the single qubit from the state vector
        num_qubits = len(circuit.all_qubits())
        qubit_state = state_vector[qubit_index::num_qubits]

        # Normalize the state to get the Bloch vector components
        theta = 2 * np.arccos(np.abs(qubit_state[0]))  # theta is the angle from the Z axis
        phi = np.angle(qubit_state[0])  # phi is the azimuthal angle in the XY plane

        # Convert the Bloch vector components to Cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        # Plot the Bloch sphere
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(0, 0, 0, x, y, z, length=1.0, normalize=True)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_title(f"Circuit {i}, Qubit {qubit_index} on Bloch Sphere")

        plt.show()

        # Clear memory after each plot to prevent memory overload
        del result, state_vector, qubit_state
        gc.collect()  # Explicit garbage collection

def eda_univariate_analysis_plot_numeric_features(num_col):
    disc_num_var = [TARGET, 'id']

    cont_num_var = []
    for i in num_col.columns:
        if i not in disc_num_var:
            cont_num_var.append(i)

    fig = plt.figure(figsize=(16,55))
    for index,col in enumerate(cont_num_var):
        plt.subplot(10,2,index+1)
        sns.distplot(num_col.loc[:,col].dropna(), kde=False)
    fig.tight_layout(pad=2.0)

def eda_univariate_analysis_plot_variables_mostly_one_value(cont_num_var,train):
    ncols = 2
    # Loop over each numeric column and plot the histograms
    for n, col in enumerate(cont_num_var):
        if n % ncols == 0:  # Create a new figure every time we reach a new row of plots
            fig, axs = plt.subplots(ncols=ncols, figsize=(35, 20))
        ax = axs[n % ncols]  # Select the appropriate subplot for the current column
        sns.histplot(data=train[train.IsLaundering == 0], x=col, kde=True, stat='density', color='b', ax=ax)
        sns.histplot(data=train[train.IsLaundering == 1], x=col, kde=True, stat='density', color='r', ax=ax)

    # Show all the plots after the loop
    plt.tight_layout()  # Adjust spacing to prevent overlapping of subplots
    plt.show()

def eda_univariate_analysis_plot_boxplot_to_identify_outliers(cont_num_var,num_col):
    fig = plt.figure(figsize=(16,35))
    for index,col in enumerate(cont_num_var):
        plt.subplot(10,4,index+1)
        sns.boxplot(y=col, data=num_col.dropna())
    fig.tight_layout(pad=2.0)

def eda_univariate_analysis_plot_countplot_to_identify_outliers(disc_num_var,num_col):
    fig = plt.figure(figsize=(75,55))
    for index,col in enumerate(disc_num_var):
        plt.subplot(8,10,index+1)
        sns.countplot(x=col, data=num_col.dropna())
    fig.tight_layout(pad=1.0)

def eda_univariate_analysis_plot_countplot_distinct_value_within_each_features(cat_col):
    fig = plt.figure(figsize=(18,35))
    for index in range(len(cat_col.columns)):
        plt.subplot(5,3,index+1)
        sns.countplot(x=cat_col.iloc[:,index], data=cat_col.dropna())
        plt.xticks(rotation=90)
    fig.tight_layout(pad=2.0)

def eda_bivariate_analysis_plot_correlation_matrix(num_col):
    plt.figure(figsize=(14,12))
    correlation = num_col.corr()
    sns.heatmap(correlation, mask = correlation <0.8, linewidth=0.5, cmap='Blues')

def eda_bivariate_analysis_plot_sactter_plot_to_identify_linear_relationship(col,train,TARGET):
    fig = plt.figure(figsize=(16,35))
    for index in range(len(col)):
        plt.subplot(4,8,index+1)
        sns.scatterplot(x=train.iloc[:,index], y=TARGET, data=train.dropna())
    fig.tight_layout(pad=2.0)

def eda_bivariate_analysis_make_mutualinformation_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def eda_bivariate_analysis_plot_mutualinformation_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

def eda_bivariate_analysis_plot_mutualinformation_catplot(X_raw_full,TARGET):
    sns.catplot(x="", y=TARGET, data=X_raw_full, kind="boxen")

def eda_bivariate_analysis_plot_mutualinformation_lmplot(feature,X_raw_full,TARGET):
    sns.lmplot(
        x=feature, y=TARGET, hue="", col="",
        data=X_raw_full, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
    )

def eda_data_inbalance_plot_pie_chart(train,labels,TARGET):
    
    fraud_or_not = train[TARGET].value_counts().tolist()
    values = [fraud_or_not[0], fraud_or_not[1]]

    fig = px.pie(values=fraud_or_not, names=labels , width=1000, height=400, color_discrete_sequence=["skyblue","black"]
                ,title="Fraud vs Genuine transactions")
    fig.show()

def eda_data_inbalance_plot_countplot(train,TARGET):
    
    plt.figure(figsize=(3,4))
    ax = sns.countplot(x=TARGET,data=train,palette="pastel")
    for i in ax.containers:
        ax.bar_label(i,)

    print('Genuine:', round(train[TARGET].value_counts()[0]/len(train) * 100,2), '% of the dataset')
    print('Frauds:', round(train[TARGET].value_counts()[1]/len(train) * 100,2), '% of the dataset')

def eda_redundant_features_plot_barplot_of_missing_values(train):
    plt.figure(figsize=(25,8))
    plt.title('Number of missing rows')
    missing_count = pd.DataFrame(train.isnull().sum(), columns=['sum']).sort_values(by=['sum'],ascending=False).head(20).reset_index()
    missing_count.columns = ['features','sum']
    sns.barplot(x='features',y='sum', data = missing_count)

def eda_redundant_features_plot_heatmap_useless_features(TARGET,num_col_TARGET):
    fig,axes = plt.subplots(1,2, figsize=(15,5))
    sns.regplot(x=num_col_TARGET['AmountReceived'], y=TARGET, data=num_col_TARGET, ax = axes[0], line_kws={'color':'black'})
    sns.regplot(x=num_col_TARGET['AmountPaid'], y=TARGET, data=num_col_TARGET, ax = axes[1], line_kws={'color':'black'})
    fig.tight_layout(pad=2.0)

def eda_dealing_outliers_plot_boxplot(train,out_col):
    fig = plt.figure(figsize=(20,5))
    for index,col in enumerate(out_col):
        plt.subplot(1,5,index+1)
        sns.boxplot(y=col, data=train)
    fig.tight_layout(pad=1.5)

def eda_plot_cluster(X, features, y, target_column, train, n_clusters=20):
    # Copy the dataset and add cluster labels
    Xy = X.copy()
    
    # Ensure the "Cluster" column exists
    Xy["Cluster"] = Xy["Cluster"].astype("category")
    
    # Add the target column to Xy
    Xy[target_column] = y
    
    # Add the original features from the train DataFrame to Xy
    Xy[features] = train[features]  # Make sure to replace 'train' with your actual DataFrame if it's different
    
    # Melt the DataFrame for seaborn
    melted_data = Xy.melt(value_vars=features, id_vars=[target_column, "Cluster"])
    
    # Plot using seaborn's relplot
    sns.relplot(
        x="value", y=target_column, hue="Cluster", col="variable",
        height=4, aspect=1, facet_kws={'sharex': False}, col_wrap=3,
        data=melted_data
    )
        
def impact_metrics(model_deep,X_test,y_test):    
    # 1) Get probabilities
    y_pred_prob = model_deep.predict(X_test).ravel()         # shape (n_samples,)
    # 2) Pick a threshold, e.g. 0.5
    y_pred = (y_pred_prob > 0.5).astype(int)

    # 3) Compute scores
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_pred_prob)

    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"ROC‑AUC:   {auc:.3f}")

def impact_get_precision_recall_curve(model_deep,y_test,y_pred_prob):

    # get precision/recall pairs for all thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)

    # find threshold that gives, say, recall ≥ 0.8
    desired_recall = 0.8
    ix = np.argmax(recalls >= desired_recall)
    print("Threshold:", thresholds[ix])
    print(" -> precision:", precisions[ix], " recall:", recalls[ix])

def impact_plot_roc_curve(model_deep,y_test,y_pred_prob):
    # ROC curve
    R = RocCurveDisplay.from_predictions(y_test, y_pred_prob)
    plt.title("ROC Curve")

    # Precision–Recall curve
    P = PrecisionRecallDisplay.from_predictions(y_test, y_pred_prob)
    plt.title("Precision–Recall Curve")


def impact_plot_confusion_matrix(y_test,predictions_deep,predictions_hybrid):

    threshold = 0.5
    # Simulated results (replace with real model outputs)
    y_true = y_test  # Actual fraud labels
    y_pred_traditional = (predictions_deep >= threshold).astype(int)  # AI model predictions
    y_pred_quantum = (predictions_hybrid >= threshold).astype(int)  # AI + Quantum predictions

    # Compute confusion matrices
    cm_traditional = confusion_matrix(y_true, y_pred_traditional)
    cm_quantum = confusion_matrix(y_true, y_pred_quantum)

    # Plot confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Traditional AI Model Confusion Matrix
    sns.heatmap(cm_traditional, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraud", "Fraud"], yticklabels=["No Fraud", "Fraud"], ax=axes[0])
    axes[0].set_title("Traditional AI Model")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # AI + Quantum Model Confusion Matrix
    sns.heatmap(cm_quantum, annot=True, fmt="d", cmap="Greens", xticklabels=["No Fraud", "Fraud"], yticklabels=["No Fraud", "Fraud"], ax=axes[1])
    axes[1].set_title("AI + Quantum Model")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()

def impact_plot_roc_curve_comparation(y_test,predictions_deep,predictions_hybrid):
    # Assuming predictions_deep and predictions_hybrid contain predicted probabilities
    fpr_deep, tpr_deep, _ = roc_curve(y_test, predictions_deep)
    fpr_quantum, tpr_quantum, _ = roc_curve(y_test, predictions_hybrid)

    roc_auc_deep = auc(fpr_deep, tpr_deep)
    roc_auc_quantum = auc(fpr_quantum, tpr_quantum)

    # Plotting both ROC curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_deep, tpr_deep, color='blue', label=f'Deep Learning (AUC = {roc_auc_deep:.3f})')
    plt.plot(fpr_quantum, tpr_quantum, color='green', label=f'Hybrid Quantum (AUC = {roc_auc_quantum:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal reference line

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def impact_plot_roc_auc_score(y_true,predictions_hybrid):
    # Calculate ROC-AUC for Deep Learning
    roc_auc = roc_auc_score(y_true, predictions_hybrid)
    print("Hybrid ROC-AUC:", roc_auc)

def impact_plot_prediction_confidence(predictions_deep, predictions_hybrid):
    plt.hist(predictions_deep, bins=50, alpha=0.5, label='Traditional')
    plt.hist(predictions_hybrid, bins=50, alpha=0.5, label='Quantum')
    plt.legend()
    plt.title("Prediction Confidence")
    plt.show()

def impact_classification_report(y_true, y_pred_traditional, y_pred_quantum, predictions_deep, predictions_hybrid):

    print("Traditional:\n", classification_report(y_true, y_pred_traditional))
    print("Quantum:\n", classification_report(y_true, y_pred_quantum))
    print(np.array_equal(predictions_deep, predictions_hybrid))  # Should be False

def featuring_engeneering_plot_skewed_target_variable_before_transformation(train,TARGET):
    plt.figure(figsize=(10,6))
    plt.title("Before transformation of Class")
    dist = sns.distplot(train[TARGET],norm_hist=False)

def featuring_engeneering_plot_skewed_target_variable_after_transformation(train,TARGET):
    plt.figure(figsize=(10,6))
    plt.title("After transformation of Class")
    dist = sns.distplot(np.log(train[TARGET]),norm_hist=False)

def featuring_engeneering_plot_variance(pca, width=8, dpi=100):
    
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    
    fig.set(figwidth=8, dpi=100)
    return axs

def featuring_engeneering_corrplot(df, method="pearson", annot=True, **kwargs):
    sns.clustermap(
        df.corr(method),
        vmin=-1.0,
        vmax=1.0,
        cmap="icefire",
        method="complete",
        annot=annot,
        **kwargs,
    )
