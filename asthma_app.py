import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans

st.title("Asthma Data Mining Interactive Demo")

# Load dataset
df = pd.read_csv("Asthma.csv")
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Filter
income_filter = st.selectbox("Filter by Income Level:", df["income_level"].unique())
filtered_df = df[df["income_level"] == income_filter]
st.write(f"Showing data for income level: **{income_filter}**")
st.dataframe(filtered_df.head())

# Preprocessing
df_encoded = df.copy()

label_cols = ["gender", "race_ethnicity", "income_level", "region", "urban_rural", "insurance_status"]
for col in label_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df[col].astype(str))

scaler = MinMaxScaler()
num_cols = ["age", "air_quality_index", "asthma_attacks_past_year", "emergency_visits", "doctor_visits"]
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# Logistic Regression
X = df_encoded.drop("asthma_current", axis=1)
y = df_encoded["asthma_current"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
st.pyplot(fig)

# ROC Curve
st.subheader("ROC Curve")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
ax2.plot([0,1], [0,1], "k--")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()
st.pyplot(fig2)

# Feature Importance
st.subheader("Feature Importance")
importance = pd.Series(model.coef_[0], index=X.columns)

fig3, ax3 = plt.subplots(figsize=(6,4))
importance.sort_values().plot(kind="barh", ax=ax3, color="purple")
st.pyplot(fig3)

# Clustering
st.subheader("K-Means Clustering")
cluster_features = ["age", "air_quality_index", "income_level",
                    "asthma_attacks_past_year", "doctor_visits"]

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_encoded[cluster_features])
df_encoded["Cluster"] = clusters

fig4, ax4 = plt.subplots()
scatter = ax4.scatter(df_encoded["air_quality_index"], df_encoded["doctor_visits"],
                      c=df_encoded["Cluster"], cmap="viridis")
ax4.set_xlabel("Air Quality Index")
ax4.set_ylabel("Doctor Visits")
st.pyplot(fig4)
