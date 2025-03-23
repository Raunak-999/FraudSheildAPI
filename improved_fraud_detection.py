import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings
import os
warnings.filterwarnings('ignore')

def load_data():
    # Get the data directory
    data_dir = os.path.join(os.getcwd(), 'New folder')
    
    # Define file paths
    train_file = os.path.join(data_dir, 'transactions_train.csv')
    test_file = os.path.join(data_dir, 'transactions_test_wo_target.csv')
    template_file = os.path.join(data_dir, 'test_submission_template.csv')
    
    if not all(os.path.exists(f) for f in [train_file, test_file, template_file]):
        raise FileNotFoundError(
            "Could not find the required CSV files. Please ensure they are in the correct location.\n"
            f"Looking in: {data_dir}\n"
            "Required files: transactions_train.csv, transactions_test_wo_target.csv, test_submission_template.csv"
        )
    
    print(f"Loading data from: {data_dir}")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    submission_template = pd.read_csv(template_file)
    
    return train_df, test_df, submission_template

def preprocess_data(df, is_training=True, y=None):
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Handle missing values
    df_processed['transaction_amount'] = df_processed['transaction_amount'].fillna(df_processed['transaction_amount'].median())
    
    # Convert date and create time features
    df_processed['transaction_date'] = pd.to_datetime(df_processed['transaction_date'])
    df_processed['transaction_hour'] = df_processed['transaction_date'].dt.hour
    df_processed['transaction_day'] = df_processed['transaction_date'].dt.day
    df_processed['transaction_dow'] = df_processed['transaction_date'].dt.dayofweek
    df_processed['transaction_weekend'] = df_processed['transaction_dow'].isin([5, 6]).astype(int)
    
    # Create cyclical time features
    df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['transaction_hour']/24)
    df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['transaction_hour']/24)
    df_processed['day_sin'] = np.sin(2 * np.pi * df_processed['transaction_day']/31)
    df_processed['day_cos'] = np.cos(2 * np.pi * df_processed['transaction_day']/31)
    
    # Handle transaction amount outliers using IQR
    if is_training:
        global amount_bounds
        Q1 = df_processed['transaction_amount'].quantile(0.25)
        Q3 = df_processed['transaction_amount'].quantile(0.75)
        IQR = Q3 - Q1
        amount_bounds = {
            'lower': Q1 - 1.5*IQR,
            'upper': Q3 + 1.5*IQR
        }
    
    df_processed['transaction_amount'] = df_processed['transaction_amount'].clip(
        lower=amount_bounds['lower'], 
        upper=amount_bounds['upper']
    )
    
    # Create amount-based features
    df_processed['amount_log'] = np.log1p(df_processed['transaction_amount'])
    
    # Categorical encoding
    categorical_cols = [
        'transaction_channel',
        'payee_id_anonymous'
    ]
    
    if is_training:
        # Calculate encoding maps during training
        global encoding_maps
        encoding_maps = {}
        df_with_target = df_processed.copy()
        df_with_target['is_fraud'] = y
        
        for col in categorical_cols:
            # Risk encoding (mean target encoding)
            encoding_maps[f'{col}_risk'] = df_with_target.groupby(col)['is_fraud'].mean()
            # Frequency encoding
            encoding_maps[f'{col}_freq'] = df_with_target[col].value_counts(normalize=True)
            # Label encoding for rare categories
            encoding_maps[f'{col}_labels'] = {v: i for i, v in enumerate(df_with_target[col].unique())}
    
    # Apply encodings
    for col in categorical_cols:
        # Risk encoding
        df_processed[f'{col}_risk'] = df_processed[col].map(encoding_maps[f'{col}_risk']).fillna(0)
        # Frequency encoding
        df_processed[f'{col}_freq'] = df_processed[col].map(encoding_maps[f'{col}_freq']).fillna(0)
        # Label encoding
        df_processed[f'{col}_label'] = df_processed[col].map(encoding_maps[f'{col}_labels']).fillna(-1)
    
    # Drop unnecessary columns
    columns_to_drop = categorical_cols + ['transaction_date']
    df_processed = df_processed.drop(columns_to_drop, axis=1)
    
    # Ensure all columns are numeric
    for col in df_processed.columns:
        if not np.issubdtype(df_processed[col].dtype, np.number):
            print(f"Warning: Converting {col} to numeric")
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
    
    # Scale numerical features
    if is_training:
        global scaler
        scaler = StandardScaler()
        numerical_cols = ['transaction_amount', 'amount_log'] + \
                        [col for col in df_processed.columns if '_risk' in col or '_freq' in col]
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    else:
        numerical_cols = ['transaction_amount', 'amount_log'] + \
                        [col for col in df_processed.columns if '_risk' in col or '_freq' in col]
        df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])
    
    return df_processed

def train_model(X, y):
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X, y)
    return model

def find_optimal_threshold(model, X_val, y_val):
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = classification_report(y_val, y_pred, output_dict=True)['1']['f1-score']
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold

def main():
    # Load data
    train_df, test_df, submission_template = load_data()
    
    print("\nClass distribution in training data:")
    print(train_df['is_fraud'].value_counts(normalize=True))
    
    # Split features and target for training
    train_features = train_df.drop(['is_fraud', 'transaction_id_anonymous'], axis=1)
    train_target = train_df['is_fraud']
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_features, train_target, test_size=0.2, random_state=42, stratify=train_target
    )
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train_processed = preprocess_data(X_train, is_training=True, y=y_train)
    X_val_processed = preprocess_data(X_val, is_training=False)
    
    # Train model
    model = train_model(X_train_processed, y_train)
    
    # Find optimal threshold using validation set
    print("\nFinding optimal threshold...")
    optimal_threshold = find_optimal_threshold(model, X_val_processed, y_val)
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Save model and preprocessing objects
    import joblib
    print("\nSaving model and preprocessing objects...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoding_maps, 'encoders.pkl')
    joblib.dump(amount_bounds, 'amount_bounds.pkl')
    joblib.dump(optimal_threshold, 'optimal_threshold.pkl')
    print("Objects saved successfully!")
    
    # Evaluate on validation set
    val_pred_proba = model.predict_proba(X_val_processed)[:, 1]
    val_pred = (val_pred_proba >= optimal_threshold).astype(int)
    
    print("\nValidation Set Metrics:")
    print(classification_report(y_val, val_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_val, val_pred_proba):.4f}")
    
    # Process test data and make predictions
    print("\nProcessing test data and making predictions...")
    test_features = test_df.drop(['transaction_id_anonymous'], axis=1)
    test_processed = preprocess_data(test_features, is_training=False)
    test_pred_proba = model.predict_proba(test_processed)[:, 1]
    test_pred = (test_pred_proba >= optimal_threshold).astype(int)
    
    # Create submission file
    submission_template['is_fraud'] = test_pred.astype(int)
    submission_file = 'improved_submission.csv'
    submission_template.to_csv(submission_file, index=False)
    
    print(f"\nSubmission file created: {submission_file}")
    print("\nPrediction distribution in test set:")
    print(pd.Series(test_pred).value_counts(normalize=True))
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train_processed.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()
