import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Streamlit App Title
st.title("🛒 Market Basket Analysis")

# Upload CSV File
uploaded_file = st.file_uploader("📂 Upload Transaction Dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview")
    st.dataframe(df.head())

    # Check if required columns exist
    if "TransactionID" in df.columns and "Item" in df.columns:
        # Convert transactions into One-Hot Encoding
        transactions = df.groupby("TransactionID")["Item"].apply(list).tolist()
        unique_items = sorted(set(item for sublist in transactions for item in sublist))
        encoded_data = pd.DataFrame([{item: (item in trans) for item in unique_items} for trans in transactions])
    else:
        st.error("CSV must contain 'TransactionID' and 'Item' columns.")
        st.stop()

    # User selects Min Support
    min_support = st.slider("📊 Select Min Support", 0.01, 0.5, 0.05)
    frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)
    
    st.write("### Frequent Itemsets")
    st.dataframe(frequent_itemsets)

    # User selects Min Confidence
    min_confidence = st.slider("📊 Select Min Confidence", 0.1, 1.0, 0.5)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    st.write("### Association Rules")
    st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

    # Visualization
    st.write("### 📈 Support vs Confidence")
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=rules, x="support", y="confidence", hue="lift", size="lift", palette="viridis")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    st.pyplot(plt)
