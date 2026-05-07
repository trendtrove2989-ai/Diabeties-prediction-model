
import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from flask import Flask, render_template, jsonify, request

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

app = Flask(__name__)

raw_df = None           # original loaded data
clean_df = None         # cleaned & encoded data
model = None            # trained logistic regression
scaler = None           # fitted StandardScaler
model_metrics = {}      # accuracy, precision, recall, f1, confusion matrix, roc
feature_names = []      # columns used for training
label_encoders = {}     # for categorical encoding
numerical_cols = []
categorical_cols = []



def load_and_clean_data():
    """Load CSV, clean, encode, normalize, and train model."""
    global raw_df, clean_df, model, scaler, model_metrics, feature_names
    global label_encoders, numerical_cols, categorical_cols
    # Note: raw_df is reassigned after cleaning so visualizations use clean data

    csv_path = os.path.join(os.path.dirname(__file__), 'Diabties Data set.csv')
    raw_df = pd.read_csv(csv_path)

    print(f"\n{'='*60}")
    print(f"  DIABETES DATASET ANALYSIS")
    print(f"{'='*60}")
    print(f"  Rows: {raw_df.shape[0]:,}  |  Columns: {raw_df.shape[1]}")
    print(f"  Target distribution:")
    print(f"    No Diabetes (0): {(raw_df['diabetes']==0).sum():,}")
    print(f"    Diabetes    (1): {(raw_df['diabetes']==1).sum():,}")
    print(f"{'='*60}\n")

    # ── Work on a copy ──
    df = raw_df.copy()

    # ── 1. Handle missing values ──
    print("[1/8] Handling missing values...")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"  Found {missing.sum()} missing values")
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
    else:
        print("  No missing values found [OK]")

    # ── 2. Remove duplicates ──
    print("[2/8] Removing duplicates...")
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"  Removed {before - after:,} duplicate rows")

    print("[3/8] Fixing unrealistic BMI values...")
    bmi_outliers = (df['bmi'] > 60).sum()
    median_bmi = df.loc[df['bmi'] <= 60, 'bmi'].median()
    df.loc[df['bmi'] > 60, 'bmi'] = median_bmi
    print(f"  Capped {bmi_outliers} BMI values > 60 to median ({median_bmi:.2f})")


    print("[4/8] Fixing contradictory diabetes labels...")
    contradictory = (df['hbA1c_level'] >= 6.5) & (df['blood_glucose_level'] >= 200) & (df['diabetes'] == 0)
    n_fixed = contradictory.sum()
    df.loc[contradictory, 'diabetes'] = 1
    print(f"  Relabeled {n_fixed} records (HbA1c>=6.5 AND glucose>=200 but diabetes=0)")

 
    print("[5/8] Cleaning infant data anomalies...")
    infant_mask = df['age'] < 2

    infant_high_hba1c = infant_mask & (df['hbA1c_level'] >= 6.0)
    n_infant_fix = infant_high_hba1c.sum()
    df.loc[infant_high_hba1c, 'hbA1c_level'] = 5.0

    infant_high_bmi = infant_mask & (df['bmi'] > 22)
    n_infant_bmi = infant_high_bmi.sum()
    infant_bmi_median = df.loc[infant_mask & (df['bmi'] <= 22), 'bmi'].median()
    if pd.notna(infant_bmi_median):
        df.loc[infant_high_bmi, 'bmi'] = infant_bmi_median
    print(f"  Fixed {n_infant_fix} infant HbA1c values, {n_infant_bmi} infant BMI values")


    print("[6/8] Capping extreme blood glucose values...")
    glucose_high = (df['blood_glucose_level'] > 300).sum()
    df.loc[df['blood_glucose_level'] > 300, 'blood_glucose_level'] = 300
    glucose_low = (df['blood_glucose_level'] < 70).sum()
    df.loc[df['blood_glucose_level'] < 70, 'blood_glucose_level'] = 70
    print(f"  Capped {glucose_high} values > 300 and {glucose_low} values < 70")

    raw_df = df.copy()

    print("[7/8] Encoding categorical variables...")

    df['gender_encoded'] = (df['gender'] == 'Male').astype(int)

    smoking_map = {
        'never': 0, 'No Info': 1, 'former': 2,
        'not current': 3, 'ever': 4, 'current': 5
    }
    df['smoking_encoded'] = df['smoking_history'].map(smoking_map)

    # Location: label encode
    le_location = LabelEncoder()
    df['location_encoded'] = le_location.fit_transform(df['location'])
    label_encoders['location'] = le_location

    print(f"  Encoded gender, smoking_history, location [OK]")

    # ── Feature selection ──
    print("  Selecting features...")
    feature_cols = [
        'age', 'bmi', 'hbA1c_level', 'blood_glucose_level',
        'hypertension', 'heart_disease',
        'gender_encoded', 'smoking_encoded',
        'race:AfricanAmerican', 'race:Asian', 'race:Caucasian',
        'race:Hispanic', 'race:Other'
    ]
    target_col = 'diabetes'
    feature_names = feature_cols

    # Identify column types for frontend
    numerical_cols = ['year', 'age', 'bmi', 'hbA1c_level', 'blood_glucose_level']
    categorical_cols = ['gender', 'location', 'smoking_history',
                        'hypertension', 'heart_disease', 'diabetes',
                        'race:AfricanAmerican', 'race:Asian', 'race:Caucasian',
                        'race:Hispanic', 'race:Other']

    X = df[feature_cols].values
    y = df[target_col].values

    # ── Normalize numerical features ──
    print("  Normalizing numerical features...")
    scaler = StandardScaler()
    # Only scale continuous features (first 4)
    continuous_idx = [0, 1, 2, 3]  # age, bmi, hbA1c, glucose
    X[:, continuous_idx] = scaler.fit_transform(X[:, continuous_idx])
    print(f"  StandardScaler applied to: age, bmi, hbA1c_level, blood_glucose_level [OK]")

    # ── 8. Handle class imbalance with SMOTE ──
    print("[8/8] Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"  Before SMOTE: {len(y)} samples")
    print(f"  After SMOTE:  {len(y_resampled)} samples")
    print(f"  Class 0: {(y_resampled==0).sum():,}  |  Class 1: {(y_resampled==1).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    print(f"\n{'-'*60}")
    print("  TRAINING LOGISTIC REGRESSION MODEL")
    print(f"{'-'*60}")

    model = LogisticRegression(
        C=1.0, max_iter=1000, solver='lbfgs', random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='accuracy')

    model_metrics = {
        'accuracy': round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'roc_auc': round(roc_auc * 100, 2),
        'cv_mean': round(cv_scores.mean() * 100, 2),
        'cv_std': round(cv_scores.std() * 100, 2),
        'confusion_matrix': cm.tolist(),
        'roc_fpr': fpr[::max(1, len(fpr)//100)].tolist(),
        'roc_tpr': tpr[::max(1, len(tpr)//100)].tolist(),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'total_samples_after_smote': len(y_resampled)
    }

    print(f"\n  Accuracy:    {acc*100:.2f}%")
    print(f"  Precision:   {prec*100:.2f}%")
    print(f"  Recall:      {rec*100:.2f}%")
    print(f"  F1 Score:    {f1*100:.2f}%")
    print(f"  ROC AUC:     {roc_auc*100:.2f}%")
    print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}")
    print(f"\n{'='*60}\n")

    # Store cleaned df for the frontend (use original columns, not encoded)
    clean_df = df

    return df


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/columns')
def get_columns():
    """Return column names grouped by type."""
    all_cols = list(raw_df.columns)
    num_cols = list(raw_df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(raw_df.select_dtypes(include=['object']).columns)
    return jsonify({
        'all': all_cols,
        'numerical': num_cols,
        'categorical': cat_cols
    })


@app.route('/api/stats')
def get_stats():
    """Descriptive statistics for every column."""
    df = raw_df.copy()
    result = {}

    for col in df.columns:
        info = {'name': col, 'dtype': str(df[col].dtype), 'missing': int(df[col].isnull().sum())}
        if df[col].dtype in ['int64', 'float64']:
            info['count'] = int(df[col].count())
            info['mean'] = round(float(df[col].mean()), 4)
            info['median'] = round(float(df[col].median()), 4)
            info['mode'] = round(float(df[col].mode().iloc[0]), 4) if not df[col].mode().empty else None
            info['std'] = round(float(df[col].std()), 4)
            info['min'] = round(float(df[col].min()), 4)
            info['max'] = round(float(df[col].max()), 4)
            info['q1'] = round(float(df[col].quantile(0.25)), 4)
            info['q3'] = round(float(df[col].quantile(0.75)), 4)
            info['iqr'] = round(float(df[col].quantile(0.75) - df[col].quantile(0.25)), 4)
            info['skewness'] = round(float(df[col].skew()), 4)
            info['kurtosis'] = round(float(df[col].kurtosis()), 4)
            # Outlier count using IQR
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
            info['outliers'] = int(outliers)
        else:
            info['count'] = int(df[col].count())
            info['unique'] = int(df[col].nunique())
            info['top'] = str(df[col].mode().iloc[0]) if not df[col].mode().empty else None
            vc = df[col].value_counts().head(10).to_dict()
            info['value_counts'] = {str(k): int(v) for k, v in vc.items()}
        result[col] = info

    return jsonify(result)


@app.route('/api/dataset-overview')
def dataset_overview():
    """High-level dataset info."""
    df = raw_df
    return jsonify({
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'column_names': list(df.columns),
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
        'missing_total': int(df.isnull().sum().sum()),
        'duplicates': int(df.duplicated().sum()),
        'memory_mb': round(df.memory_usage(deep=True).sum() / 1e6, 2),
        'diabetes_distribution': {
            'no_diabetes': int((df['diabetes'] == 0).sum()),
            'diabetes': int((df['diabetes'] == 1).sum()),
            'diabetes_pct': round((df['diabetes'] == 1).mean() * 100, 2)
        }
    })


@app.route('/api/correlation')
def get_correlation():
    """Correlation matrix for numerical columns."""
    df = raw_df.select_dtypes(include=[np.number])
    corr = df.corr().round(4)
    return jsonify({
        'columns': list(corr.columns),
        'values': corr.values.tolist()
    })


@app.route('/api/histogram')
def get_histogram():
    """Histogram data for a given column."""
    col = request.args.get('column', 'age')
    if col not in raw_df.columns:
        return jsonify({'error': f'Column {col} not found'}), 400

    series = raw_df[col].dropna()

    if series.dtype in ['int64', 'float64']:
        return jsonify({
            'column': col,
            'type': 'numerical',
            'values': series.tolist(),
            'mean': round(float(series.mean()), 2),
            'median': round(float(series.median()), 2),
            'std': round(float(series.std()), 2)
        })
    else:
        vc = series.value_counts()
        return jsonify({
            'column': col,
            'type': 'categorical',
            'labels': vc.index.tolist(),
            'counts': vc.values.tolist()
        })


@app.route('/api/scatter')
def get_scatter():
    """Scatter plot data for X vs Y column."""
    x_col = request.args.get('x', 'age')
    y_col = request.args.get('y', 'bmi')
    color_col = request.args.get('color', 'diabetes')

    if x_col not in raw_df.columns or y_col not in raw_df.columns:
        return jsonify({'error': 'Column not found'}), 400

    # Sample for performance if > 5000 rows
    df = raw_df
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

    result = {
        'x_col': x_col,
        'y_col': y_col,
        'x_values': df[x_col].tolist(),
        'y_values': df[y_col].tolist()
    }

    if color_col in df.columns:
        result['color_col'] = color_col
        result['color_values'] = df[color_col].tolist()

    return jsonify(result)


@app.route('/api/bar')
def get_bar():
    """Bar chart data — categorical distribution or mean of numerical by category."""
    col = request.args.get('column', 'gender')
    group_by = request.args.get('group_by', None)

    if col not in raw_df.columns:
        return jsonify({'error': 'Column not found'}), 400

    if group_by and group_by in raw_df.columns:
        # Mean of numerical column grouped by categorical
        grouped = raw_df.groupby(col)[group_by].mean().round(4)
        return jsonify({
            'column': col,
            'group_by': group_by,
            'labels': grouped.index.tolist(),
            'values': grouped.values.tolist(),
            'chart_type': 'grouped'
        })
    else:
        vc = raw_df[col].value_counts()
        return jsonify({
            'column': col,
            'labels': [str(l) for l in vc.index.tolist()],
            'values': vc.values.tolist(),
            'chart_type': 'simple'
        })


@app.route('/api/boxplot')
def get_boxplot():
    """Box plot data for a numerical column, optionally grouped."""
    col = request.args.get('column', 'bmi')
    group_by = request.args.get('group_by', 'diabetes')

    if col not in raw_df.columns:
        return jsonify({'error': 'Column not found'}), 400

    if group_by and group_by in raw_df.columns:
        groups = {}
        for name, group in raw_df.groupby(group_by):
            groups[str(name)] = group[col].dropna().tolist()
        return jsonify({
            'column': col,
            'group_by': group_by,
            'groups': groups
        })
    else:
        return jsonify({
            'column': col,
            'values': raw_df[col].dropna().tolist()
        })


@app.route('/api/pairwise')
def get_pairwise():
    """Get data for pairwise comparison of two columns with diabetes coloring."""
    x_col = request.args.get('x', 'age')
    y_col = request.args.get('y', 'bmi')

    if x_col not in raw_df.columns or y_col not in raw_df.columns:
        return jsonify({'error': 'Column not found'}), 400

    df = raw_df.sample(min(3000, len(raw_df)), random_state=42)

    diabetic = df[df['diabetes'] == 1]
    non_diabetic = df[df['diabetes'] == 0]

    return jsonify({
        'x_col': x_col,
        'y_col': y_col,
        'diabetic': {
            'x': diabetic[x_col].tolist(),
            'y': diabetic[y_col].tolist()
        },
        'non_diabetic': {
            'x': non_diabetic[x_col].tolist(),
            'y': non_diabetic[y_col].tolist()
        }
    })


@app.route('/api/model-performance')
def get_model_performance():
    """Return model evaluation metrics."""
    return jsonify(model_metrics)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict diabetes risk for given patient data."""
    try:
        data = request.get_json()

        # Build feature vector in same order as training
        age = float(data.get('age', 40))
        bmi = float(data.get('bmi', 27))
        hba1c = float(data.get('hbA1c_level', 5.5))
        glucose = float(data.get('blood_glucose_level', 130))
        hypertension = int(data.get('hypertension', 0))
        heart_disease = int(data.get('heart_disease', 0))
        gender = int(data.get('gender', 0))  # 0=Female, 1=Male
        smoking = int(data.get('smoking', 0))  # 0-5 scale

        # Race features
        race_aa = int(data.get('race_african_american', 0))
        race_as = int(data.get('race_asian', 0))
        race_ca = int(data.get('race_caucasian', 0))
        race_hi = int(data.get('race_hispanic', 0))
        race_ot = int(data.get('race_other', 0))

        features = np.array([[age, bmi, hba1c, glucose,
                              hypertension, heart_disease,
                              gender, smoking,
                              race_aa, race_as, race_ca, race_hi, race_ot]])

        # Scale continuous features
        features[:, :4] = scaler.transform(features[:, :4])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        return jsonify({
            'prediction': int(prediction),
            'probability_no_diabetes': round(float(probability[0]) * 100, 2),
            'probability_diabetes': round(float(probability[1]) * 100, 2),
            'risk_level': 'High' if probability[1] > 0.7 else ('Medium' if probability[1] > 0.4 else 'Low')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/feature-importance')
def feature_importance():
    """Return logistic regression coefficients as feature importance."""
    coeffs = model.coef_[0]
    importance = sorted(
        zip(feature_names, coeffs.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    return jsonify({
        'features': [f[0] for f in importance],
        'coefficients': [round(f[1], 4) for f in importance],
        'abs_coefficients': [round(abs(f[1]), 4) for f in importance]
    })


@app.route('/api/distribution-by-target')
def distribution_by_target():
    """Distribution of a numerical column split by diabetes status."""
    col = request.args.get('column', 'age')
    if col not in raw_df.columns:
        return jsonify({'error': 'Column not found'}), 400

    diabetic = raw_df[raw_df['diabetes'] == 1][col].dropna().tolist()
    non_diabetic = raw_df[raw_df['diabetes'] == 0][col].dropna().tolist()

    return jsonify({
        'column': col,
        'diabetic': diabetic,
        'non_diabetic': non_diabetic
    })


@app.route('/api/confidence-interval')
def confidence_interval():
    """Confidence interval for a numerical column (inspired by R t.test)."""
    col = request.args.get('column', 'age')
    conf_level = float(request.args.get('confidence', 0.95))
    if col not in raw_df.columns:
        return jsonify({'error': 'Column not found'}), 400

    data = raw_df[col].dropna()
    n = len(data)
    mean = float(data.mean())
    std = float(data.std())
    se = std / np.sqrt(n)

    # t-test confidence interval
    from scipy.stats import t as t_dist
    alpha = 1 - conf_level
    t_crit = float(t_dist.ppf(1 - alpha/2, df=n-1))
    ci_lower = round(mean - t_crit * se, 4)
    ci_upper = round(mean + t_crit * se, 4)

    return jsonify({
        'column': col,
        'confidence_level': conf_level,
        'n': n,
        'mean': round(mean, 4),
        'std': round(std, 4),
        'se': round(se, 4),
        't_critical': round(t_crit, 4),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'margin_of_error': round(t_crit * se, 4)
    })


@app.route('/api/density')
def density_estimation():
    """Probability density estimation for a numerical column (inspired by R geom_density)."""
    col = request.args.get('column', 'age')
    if col not in raw_df.columns:
        return jsonify({'error': 'Column not found'}), 400

    data = raw_df[col].dropna().values
    from scipy.stats import gaussian_kde, norm, shapiro
    kde = gaussian_kde(data)
    x_range = np.linspace(float(data.min()), float(data.max()), 200)
    density = kde(x_range)

    # Normal distribution fit for comparison
    mu, sigma = norm.fit(data)
    normal_pdf = norm.pdf(x_range, mu, sigma)

    # Shapiro-Wilk test for normality (sample if too large)
    sample = data if len(data) <= 5000 else np.random.choice(data, 5000, replace=False)
    try:
        shapiro_stat, shapiro_p = shapiro(sample)
    except Exception:
        shapiro_stat, shapiro_p = 0, 0

    return jsonify({
        'column': col,
        'x': x_range.tolist(),
        'density': density.tolist(),
        'normal_pdf': normal_pdf.tolist(),
        'mu': round(float(mu), 4),
        'sigma': round(float(sigma), 4),
        'shapiro_stat': round(float(shapiro_stat), 4),
        'shapiro_p': round(float(shapiro_p), 6),
        'is_normal': bool(shapiro_p > 0.05)
    })


@app.route('/api/regression-line')
def regression_line():
    """Linear regression between two columns with prediction (inspired by R lm)."""
    x_col = request.args.get('x', 'age')
    y_col = request.args.get('y', 'bmi')
    pred_val = request.args.get('predict', None)

    if x_col not in raw_df.columns or y_col not in raw_df.columns:
        return jsonify({'error': 'Column not found'}), 400

    df_clean = raw_df[[x_col, y_col]].dropna()
    x_data = df_clean[x_col].values
    y_data = df_clean[y_col].values

    # Fit linear regression
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)

    # Generate regression line points
    x_line = np.linspace(float(x_data.min()), float(x_data.max()), 100)
    y_line = slope * x_line + intercept

    # Sample scatter data for performance
    if len(x_data) > 3000:
        idx = np.random.choice(len(x_data), 3000, replace=False)
        x_scatter = x_data[idx].tolist()
        y_scatter = y_data[idx].tolist()
    else:
        x_scatter = x_data.tolist()
        y_scatter = y_data.tolist()

    result = {
        'x_col': x_col,
        'y_col': y_col,
        'slope': round(float(slope), 6),
        'intercept': round(float(intercept), 4),
        'r_squared': round(float(r_value**2), 4),
        'r_value': round(float(r_value), 4),
        'p_value': float(p_value),
        'std_err': round(float(std_err), 6),
        'x_scatter': x_scatter,
        'y_scatter': y_scatter,
        'x_line': x_line.tolist(),
        'y_line': y_line.tolist(),
        'equation': f'{y_col} = {slope:.4f} * {x_col} + {intercept:.4f}'
    }

    # Prediction at a specific value
    if pred_val is not None:
        try:
            pv = float(pred_val)
            predicted = slope * pv + intercept
            result['prediction'] = {
                'x_value': pv,
                'y_predicted': round(float(predicted), 4)
            }
        except ValueError:
            pass

    return jsonify(result)


@app.route('/api/cleaning-summary')
def cleaning_summary():
    """Summary of all data cleaning operations performed."""
    return jsonify({
        'steps': [
            {'step': 1, 'name': 'Handle Missing Values', 'description': 'Fill NaN with column median'},
            {'step': 2, 'name': 'Remove Duplicates', 'description': 'Removed exact duplicate rows'},
            {'step': 3, 'name': 'Fix Unrealistic BMI', 'description': 'Capped BMI > 60 to median (medically impossible)'},
            {'step': 4, 'name': 'Fix Contradictory Labels', 'description': 'Relabeled diabetes=0 to 1 when HbA1c >= 6.5 AND glucose >= 200 (meets diagnostic criteria)'},
            {'step': 5, 'name': 'Clean Infant Anomalies', 'description': 'Reset adult-level HbA1c and BMI for infants (age < 2)'},
            {'step': 6, 'name': 'Cap Extreme Glucose', 'description': 'Capped blood glucose to 70-300 range'},
            {'step': 7, 'name': 'Encode Categoricals', 'description': 'Binary/ordinal encoding for gender, smoking, location'},
            {'step': 8, 'name': 'SMOTE Rebalancing', 'description': 'Oversampled minority class for unbiased training'}
        ]
    })



# 1. Call this OUTSIDE the main block so Gunicorn runs it when importing the app
load_and_clean_data()

# 2. This block will now only be used if you run it locally for testing
if __name__ == '__main__':
    # Use the port assigned by the host, default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)