from pathlib import Path
from loguru import logger
from tqdm import tqdm
from anti_money_laundering.config import PROCESSED_DATA_DIR
import cirq
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import SGDOneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pandas.api.types import CategoricalDtype
from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
import seaborn as sns

def transformFeaturesScalingByPipeline(train):

    features_num = train.select_dtypes(exclude=['object']).copy().columns
    features_cat = train.select_dtypes(include=['object']).copy().columns
    
    transformer_num = make_pipeline(
        SimpleImputer(strategy="constant"),  # Handle missing values
        RobustScaler()  # Scale numerical features using RobustScaler
        #StandardScaler(),        
    )
    
    transformer_cat = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="NA"),  # Handle missing categorical values
        OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Convert categorical features to dense matrix
    )
    
    preprocessor = make_column_transformer(
        (transformer_num, features_num),  # Apply numerical transformations to numeric columns
        (transformer_cat, features_cat),   # Apply categorical transformations to categorical columns
        remainder='passthrough',  # Pass through any columns not specified for transformation
        # sparse_threshold=0.3  # Ensures final output stays sparse if enough sparse parts
    )
    return preprocessor

def transform_raw_data(X,y):
    # stratify - make sure classes are evenlly represented across splits
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.75, random_state=42)
    return X_train, X_valid, y_train, y_valid

def transform_in_batch(X, preprocessor, batch_size=50):
    batches = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        transformed = preprocessor.transform(X[start:end])
        batches.append(transformed)
    return np.vstack(batches)

def transform_applying_scalar_transform(X): 
    # # applying scalar, transform
    preprocessor = transformFeaturesScalingByPipeline(X_train)
    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    X_test = preprocessor.transform(X_test)
    return X_train, X_valid, X_test

def transform_applying_tuncatedSVD(X_train, X_valid, X_test, columns=10):
    svd = TruncatedSVD(n_components=columns, random_state=42)
    X_train = svd.fit_transform(X_train)
    X_valid = svd.transform(X_valid)
    X_test = svd.transform(X_test)
    return X_train, X_valid, X_test

def convert_to_circuit(x):
    """Convert classical feature vector x into a quantum circuit."""
    if len(x) == 0:
        raise ValueError("Input feature vector x is empty.")
        
    qubits = [cirq.GridQubit(0, i) for i in range(len(x))]
    circuit = cirq.Circuit()

    for i, qubit in enumerate(qubits):
        try:
            value = float(x[i])  # Ensure float
            circuit.append(cirq.rx(value * np.pi)(qubit))  # Encode data
        except Exception as e:
            raise ValueError(f"Error converting feature at index {i}: {x[i]} (type: {type(x[i])})") from e
    return circuit

def data_sample(TARGET,train,test,SAMPLE):

    legit = train[train[TARGET]==0]    
    legit_test = test[test[TARGET]==0]

    fraud = train[train[TARGET]==1]
    fraud_test = test[test[TARGET]==1]

    fraud = fraud.sample(SAMPLE)
    fraud_test = fraud_test.sample(SAMPLE)

    legit_sample = legit.sample(SAMPLE)
    legit_sample_test = legit_test.sample(SAMPLE)

    train = pd.concat([legit_sample,fraud],axis=0)
    test = pd.concat([legit_sample_test,fraud_test],axis=0)

    return train,test

def drop_redundant_features(train,features_num):
    train.drop(features_num, axis=1, inplace=True)
    return train

def drop_redundant_features_useless_features(train,test,TARGET,correlation,selected_features):

    correlation[[TARGET]].sort_values([TARGET], ascending=False).tail(10)

    correlation.columns.drop(['id'])

    for col in train.columns:
        if col in selected_features:
              if col != 'id':
                    if col != TARGET:
                        train[col] = train[col].apply(lambda x: 0)
                        test[col] = test[col].apply(lambda x: 0)
                        train.drop([col], axis=1, inplace=True)
                        test.drop([col], axis=1, inplace=True)
    return train,test

def drop_redundant_categorical_features_that_have_mostly_one_value(train):
    cat_col = train.select_dtypes(include=['object']).columns
    overfit_cat = []
    for i in cat_col:
        counts = train[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(train) * 100 > 96:
            overfit_cat.append(i)

    overfit_cat = list(overfit_cat)
    print(overfit_cat)

    train = train.drop(overfit_cat, axis=1)
    return train,overfit_cat

def drop_redundant_numerical_features_that_have_mostly_one_value(train):
    num_col = train.select_dtypes(exclude=['object']).copy()
    overfit_num = []
    for i in num_col:
        counts = train[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(train) * 100 > 96:
            overfit_num.append(i)

    overfit_num = list(overfit_num)
    print(overfit_num)
    
    train = train.drop(overfit_num, axis=1)
    return train,overfit_num

def drop_redundant_numerical_features_that_have_mostly_one_value(train):
    print('Number of duplicated values in dataset: ', train.duplicated().sum())
    train.drop_duplicates(inplace=True)
    return train

def drop_outliers_using_isolation_forest(data,TARGET):

    svm = SGDOneClassSVM(nu=0.55)
    yhat = svm.fit_predict(data.drop('id', axis=1))
    mask = yhat != -1
    data_SVM = data.loc[mask, :].reset_index(drop=True)   

    iso = IsolationForest(random_state=0)
    yhat = iso.fit_predict(data_SVM.drop('id', axis=1))
    mask = yhat != -1
    data_ISO = data_SVM.loc[mask, :].reset_index(drop=True)

    lof = LocalOutlierFactor(n_neighbors=3, contamination=0.1)
    yhat = lof.fit_predict(data_ISO.drop(['id', TARGET], axis=1))
    mask = yhat != -1
    data_final = data_ISO.loc[mask, :].reset_index(drop=True)

    return data_final

def drop_outliers_by_treshold(train):
    train = train.drop(train[train[''] > 200].index)
    train = train.drop(train[train[''] > 100000].index)
    train = train.drop(train[train[''] > 4000].index)
    train = train.drop(train[train[''] > 5000].index)
    train = train.drop(train[train[''] > 4000].index)

def dealing_missing_values(train):
    pd.DataFrame(train.isnull().sum(), columns=['sum']).sort_values(by=['sum'],ascending=False).head(15)

def dealing_missing_values_fill_ordinal_features_with_na(train,ord_col):
    train[ord_col] = train[ord_col].fillna("NA")

def dealing_missing_values_fill_categorical_features_with_most_frequent_value(train,cat_col):
    for i in cat_col:
        train[cat_col] = train.groupby(i)[cat_col].transform(lambda x: x.fillna(x.mode()[0]))
    return train

def dealing_missing_values_fill_categorical_features_with_mean_value(train,num_col):
    for i in num_col:
        print("Mean of " +i+ ":", train[i].mean())

    num_col.info()
    print(train[num_col].dtype)
    print(train[num_col].isna().sum())  # Shows the number of NaN values 

    train[num_col] = train[num_col] = train[num_col].fillna(train[num_col].mean())
    return train

def dealing_missing_values_fill_ordinal_features_with_string_to_value(train):
    
    # ordinal_map = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA':0}
    # fintype_map = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'NA': 0}
    # expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
    # fence_map = {'GdPrv': 4,'MnPrv': 3,'GdWo': 2, 'MnWw': 1,'NA': 0}

    # for col in ord_col:
    #     train[col] = train[col].map(ordinal_map)
        
    # fin_col = ['','']
    # for col in fin_col:
    #     train[col] = train[col].map(fintype_map)

    # train[''] = train[''].map(expose_map)
    # train[''] = train[''].map(fence_map)
    return train

def scaling_manualy(train,test):
    # rscale = RobustScaler()
    # train['']=rscale.fit_transform(train[''].values.reshape(-1,1))

    #OR

    # cols = train.select_dtypes(np.number).drop(['id', TARGET], axis=1).columns

    # for i in cols:
    #     train[i]=rscale.fit_transform(train[i].values.reshape(-1,1))
        
    #OR
        
    # cols = train.select_dtypes(np.number).drop(['id', TARGET], axis=1).columns
    # transformer = RobustScaler().fit(train[cols])
    # train[cols] = transformer.transform(train[cols])
    return train,test

def mathematical_transforms(train):
    X = pd.DataFrame()
    #...
    return X

def interactions(train):
    X = pd.get_dummies(train.colmn, prefix="")
    X = X.mul(train.colmn, axis=0)
    return X

def counts(train):
    X = pd.DataFrame()
    X[""] = train[[
        "",
        "",
    ]].gt(0.0).sum(axis=1)
    return X

def break_down(train):
    X = pd.DataFrame()
    X[""] = train.colmn.str.split("_", n=1, expand=True)[0]
    return X

def group_transforms(train):
    X = pd.DataFrame()
    X[""] = train.groupby("")[""].transform("median")
    return X

def label_encode(train):
    X = train.copy()
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    return X

cluster_features = ['FromBank', 'ToBank', 'AmountReceived', 'AmountPaid', 'ReceivingCurrency',
       'PaymentCurrency', 'PaymentFormat']


def cluster_labels(df, features, n_clusters=20):
    # Copy the DataFrame to avoid modifying the original
    X = df.copy()
    
    # Select only the specified features
    X_scaled = X.loc[:, features]
    
    # Standardize the features
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    
    # Get cluster labels
    X_new = pd.DataFrame()
    X_new["Cluster"] = kmeans.fit_predict(X_scaled)
    
    return X_new


def cluster_distance(df, features, n_clusters=20):
    # Copy the DataFrame
    X = df.copy()
    
    # Select and standardize features
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    
    # Get distance to centroids (use the transformed data, distance to centroids is the output of fit_transform)
    X_cd = kmeans.fit_transform(X_scaled)
    
    # Create DataFrame for the centroid distances
    X_cd = pd.DataFrame(X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])])
    
    return X_cd

def apply_pca(X, standardize=True):
    # Standardize
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings

# def pca_inspired(df):
#     X = pd.DataFrame()
#     X["Amount"] = df.Amount + df.
#     X["Amount"] = df. * df.
#     return X

def pca_components(df, pca_features):
    X = df.loc[:, pca_features]
    _, X_pca, _ = apply_pca(X)
    return X_pca

# def indicate_outliers(df):
#     X_new = pd.DataFrame()
#     X_new["Outlier"] = (df. == "") & (df. == "")
#     return X_new

class CrossFoldEncoder:
    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs  # keyword arguments for the encoder
        self.cv_ = KFold(n_splits=5)

    # Fit an encoder on one split and transform the feature on the
    # other. Iterating over the splits in all folds gives a complete
    # transformation. We also now have one trained encoder on each
    # fold.
    def fit_transform(self, X, y, cols):
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            fitted_encoder.fit(
                X.iloc[idx_encode, :], y.iloc[idx_encode],
            )
            X_encoded.append(fitted_encoder.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(fitted_encoder)
        X_encoded = pd.concat(X_encoded)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded

    # To transform the test data, average the encodings learned from
    # each fold.
    def transform(self, X):
        from functools import reduce

        X_encoded_list = []
        for fitted_encoder in self.fitted_encoders_:
            X_encoded = fitted_encoder.transform(X)
            X_encoded_list.append(X_encoded[self.cols_])
        X_encoded = reduce(
            lambda x, y: x.add(y, fill_value=0), X_encoded_list
        ) / len(X_encoded_list)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded


