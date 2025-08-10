import os
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    base_dir = os.path.join("D:", os.sep, "Projects", "SmartFit-SmartDiet", "data")
    diet_data_path = os.path.join(base_dir, "processed", "diet_recommendations.csv")
    cluster_data_path = os.path.join(base_dir, "processed", "pamap2_features_clustered.csv")

    df_diet = pd.read_csv(diet_data_path)
    df_cluster = pd.read_csv(cluster_data_path)

    return df_diet, df_cluster

@st.cache_resource
def load_models():
    model_dir = os.path.join("D:", os.sep, "Projects", "SmartFit-SmartDiet", "models")
    diet_model = joblib.load(os.path.join(model_dir, 'best_rf_diet_adjustment_model.joblib'))
    cluster_model = joblib.load(os.path.join(model_dir, 'kmeans.joblib'))
    # Load workout model similarly if available

    return diet_model, cluster_model

df_diet, df_cluster = load_data()

user_ids = df_diet['User_ID'].unique()
selected_user = st.sidebar.selectbox("Select User ID", user_ids)

# Filter data for selected user
user_diet_data = df_diet[df_diet['User_ID'] == selected_user]
user_cluster_data = df_cluster[df_cluster['User_ID'] == selected_user]

st.title(f"SmartFit + SmartDiet Dashboard for User {selected_user}")

st.subheader("User Profile & Cluster")
if not user_cluster_data.empty:
    cluster_label = user_cluster_data.iloc[-1]['cluster_label']  # latest cluster
    st.write(f"Current Cluster: {cluster_label}")
else:
    st.write("Cluster data not available for this user.")

st.subheader("Latest Diet Recommendation")
if not user_diet_data.empty:
    latest_rec = user_diet_data.iloc[-1]['diet_recommendation']
    st.write(latest_rec)
else:
    st.write("Diet recommendations not available.")

st.subheader("Trends Over Last 7 Days")

recent_diet = user_diet_data.tail(7)
if not recent_diet.empty:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(recent_diet['Date'], recent_diet['calorie_adjustment_pred'], marker='o')
    ax.set_title("Predicted Calorie Adjustment")
    ax.set_xlabel("Date")
    ax.set_ylabel("Calories (kcal)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # You can add more charts for macros or activity
else:
    st.write("No recent data available.")
