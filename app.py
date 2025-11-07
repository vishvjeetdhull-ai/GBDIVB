
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Universal Bank - Personal Loan Insights")

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("UniversalBank_sample.csv")
    return df

def preprocess(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lbl = None
    for c in df.columns:
        if c.lower().replace(" ","") in ("personalloan",):
            lbl = c
            break
    if lbl is None and 'Personal Loan' in df.columns:
        lbl = 'Personal Loan'
    if lbl is None:
        raise ValueError("Couldn't find 'Personal Loan' column.")
    df = df.rename(columns={lbl:'Personal_Loan'})
    for c in list(df.columns):
        if c.lower() in ('id','zip','zipcode','zip code'):
            df = df.drop(columns=[c], errors='ignore')
    df['Personal_Loan'] = df['Personal_Loan'].astype(int)
    return df

def train_models(X_train, y_train, random_state=42):
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=random_state)
    }
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted

def compute_metrics(model, X_train, y_train, X_test, y_test):
    y_tr_pred = model.predict(X_train)
    y_te_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_te_proba = model.predict_proba(X_test)[:,1]
    else:
        y_te_proba = model.decision_function(X_test)
    return {
        'Train_Accuracy': accuracy_score(y_train, y_tr_pred),
        'Test_Accuracy': accuracy_score(y_test, y_te_pred),
        'Precision': precision_score(y_test, y_te_pred, zero_division=0),
        'Recall': recall_score(y_test, y_te_pred, zero_division=0),
        'F1_Score': f1_score(y_test, y_te_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_te_proba)
    }

def plot_roc_all(models, X_test, y_test):
    fig = go.Figure()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:,1]
        else:
            proba = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(title="ROC Curves - All Models", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig

def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

st.title("Universal Bank — Personal Loan Prediction & Insights")

uploaded = st.sidebar.file_uploader("Upload CSV", type=['csv'])
if st.sidebar.button("Load sample data"):
    uploaded = None

try:
    df_raw = load_data(uploaded)
    df = preprocess(df_raw)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

tabs = st.tabs(["Overview & Charts", "Train & Evaluate Models", "Predict & Download", "Data"])

with tabs[0]:
    st.header("Customer insights — 5 charts")
    age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
    age_range = st.slider("Age range", age_min, age_max, (age_min, age_max))
    df_f = df[(df['Age']>=age_range[0]) & (df['Age']<=age_range[1])]

    # Chart A: Conversion by Age bins
    df_f['age_bin'] = pd.cut(df_f['Age'], bins=5)
    conv_age = df_f.groupby('age_bin')['Personal_Loan'].mean().reset_index()
    figA = px.bar(conv_age, x='age_bin', y='Personal_Loan', title='Conversion rate by Age bin')
    st.plotly_chart(figA, use_container_width=True)

    # Chart B: Income vs CCAvg scatter with color by Personal Loan
    figB = px.scatter(df_f, x='Income', y='CCAvg', color='Personal_Loan', hover_data=['Education','Family'], title='Income vs CCAvg by Personal Loan')
    st.plotly_chart(figB, use_container_width=True)

    # Chart C: Education vs Conversion Rate with counts
    edu = df_f.groupby('Education').agg(conv_rate=('Personal_Loan','mean'), count=('Personal_Loan','size')).reset_index()
    figC = px.bar(edu, x='Education', y='conv_rate', text='count', title='Conversion rate by Education (counts shown)')
    st.plotly_chart(figC, use_container_width=True)

    # Chart D: KMeans segmentation (Income, CCAvg, Age) and cluster conversion
    if len(df_f)>=10:
        k = st.slider("K (clusters)", 2, 6, 3)
        km = KMeans(n_clusters=k, random_state=42)
        seg = df_f[['Income','CCAvg','Age']].fillna(0)
        labels = km.fit_predict(seg)
        df_f2 = df_f.copy()
        df_f2['cluster'] = labels
        cluster_conv = df_f2.groupby('cluster')['Personal_Loan'].mean().reset_index()
        figD = px.bar(cluster_conv, x='cluster', y='Personal_Loan', title='Conversion rate by cluster')
        st.plotly_chart(figD, use_container_width=True)

    # Chart E: Absolute correlation with Personal_Loan
    corr = df.corr()['Personal_Loan'].drop('Personal_Loan').abs().sort_values(ascending=False).reset_index().rename(columns={'index':'feature','Personal_Loan':'abs_corr'})
    figE = px.bar(corr, x='feature', y='abs_corr', title='Absolute correlation with Personal Loan')
    st.plotly_chart(figE, use_container_width=True)

with tabs[1]:
    st.header("Train & Evaluate")
    X = df.drop(columns=['Personal_Loan'])
    y = df['Personal_Loan']
    test_size = st.slider("Test size (%)", 10, 40, 30)
    if st.button("Train models"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42, stratify=y)
        models = train_models(X_train, y_train)
        metrics_table = []
        for name, model in models.items():
            m = compute_metrics(model, X_train, y_train, X_test, y_test)
            row = {'Algorithm':name}
            row.update(m)
            metrics_table.append(row)
        mdf = pd.DataFrame(metrics_table).set_index('Algorithm')
        st.dataframe(mdf.style.format("{:.4f}"))
        st.plotly_chart(plot_roc_all(models, X_test, y_test), use_container_width=True)
        st.session_state['models'] = models
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test

with tabs[2]:
    st.header("Predict & Download")
    up = st.file_uploader("Upload new CSV for prediction", type=['csv'], key='pred')
    if st.button("Predict now"):
        if up is None:
            st.error("Upload file to predict.")
        else:
            nd = pd.read_csv(up)
            try:
                ndp = preprocess(nd)
            except Exception as e:
                st.error(f"Preprocess error: {e}")
                st.stop()
            if 'models' not in st.session_state:
                st.info("Models not trained in session — training now on full data.")
                models = train_models(X, y)
            else:
                models = st.session_state['models']
            model = models.get('Random Forest') or list(models.values())[0]
            X_new = ndp.drop(columns=['Personal_Loan'], errors='ignore').fillna(0)
            preds = model.predict(X_new)
            ndp['Predicted_Personal_Loan'] = preds.astype(int)
            st.dataframe(ndp.head())
            st.download_button("Download predictions CSV", data=to_csv_bytes(ndp), file_name="predictions_with_label.csv", mime="text/csv")

with tabs[3]:
    st.header("Data")
    st.dataframe(df.head(200))
    st.download_button("Download data CSV", data=to_csv_bytes(df), file_name="UniversalBank_current.csv", mime="text/csv")
