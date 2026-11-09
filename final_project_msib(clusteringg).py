import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from tabulate import tabulate

# **Langkah 1: Input API Key**
st.title("Analisis Data Transaksi Coffee Shop dengan Clustering dan AI")
st.header("Langkah 1: Masukkan API Key")
api_key = st.text_input("Masukkan API Key untuk Google Generative AI:", type="password")

if not api_key:
    st.warning("Silakan masukkan API Key untuk melanjutkan.")
else:
    # **Langkah 2: Upload Dataset**
    st.header("Langkah 2: Unggah Dataset")
    uploaded_file = st.file_uploader("Unggah dataset Anda (CSV)", type="csv")

    if uploaded_file:
        # Membaca dataset
        df = pd.read_csv(uploaded_file)
        st.write("Dataset berhasil dimuat. Berikut adalah beberapa baris pertama:")
        st.dataframe(df.head())

        # **Langkah 3: Preprocessing Data**
        st.header("Langkah 3: Preprocessing Data")
        df = df.drop_duplicates().dropna()

        if 'transaction_date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
            df['month'] = df['transaction_date'].dt.month
        
        df['total_sales'] = df['unit_price'] * df['total_qty']
        
        # Normalisasi data
        numeric_features = ['unit_price', 'total_qty', 'total_sales']
        scaler = MinMaxScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        st.write("Data setelah preprocessing:")
        st.dataframe(df.head())

        # **Langkah 4: Evaluasi Silhouette Score**
        st.header("Langkah 4: Evaluasi Silhouette Score")
        silhouette_scores = []
        for n_clusters in range(2, 16):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(df[numeric_features])
            silhouette_avg = silhouette_score(df[numeric_features], labels)
            silhouette_scores.append({'Jumlah Cluster': n_clusters, 'Silhouette Score': silhouette_avg})
        silhouette_df = pd.DataFrame(silhouette_scores)
        st.write("Silhouette Scores untuk setiap jumlah cluster:")
        st.dataframe(silhouette_df)

        # Visualisasi Silhouette Score
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=silhouette_df, x='Jumlah Cluster', y='Silhouette Score', marker='o')
        ax.set_title("Evaluasi Silhouette Score")
        ax.set_xlabel("Jumlah Cluster")
        ax.set_ylabel("Silhouette Score")
        st.pyplot(fig)

        # **Langkah 5: Clustering dengan PCA**
        st.header("Langkah 5: Clustering dengan PCA")
        n_clusters = st.slider("Pilih jumlah cluster (2–15):", min_value=2, max_value=15, value=3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Geser label cluster agar dimulai dari 2
        df['cluster'] = kmeans.fit_predict(df[numeric_features]) + 2

        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(df[numeric_features])
        df['pca_1'] = pca_features[:, 0]
        df['pca_2'] = pca_features[:, 1]

        # Visualisasi PCA
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='cluster', palette='viridis', ax=ax)
        ax.set_title(f'Visualisasi Cluster dengan {n_clusters} Cluster (PCA)')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        st.pyplot(fig)

        # **Langkah 6: Interpretasi Cluster dengan Chatbot**
        st.header("Langkah 6: Interpretasi AI")
        cluster_summary = df.groupby('cluster')[numeric_features].mean().reset_index()
        st.write(f"Ringkasan rata-rata per cluster untuk {n_clusters} cluster:")
        st.dataframe(cluster_summary)

        # Siapkan prompt untuk chatbot
        contexts = tabulate(cluster_summary, headers='keys', tablefmt='grid', showindex=False)
        question = st.text_input("Ajukan pertanyaan tentang hasil clustering (misal: Apa karakteristik tiap cluster?)")

        if question:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key, temperature=0.7)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Anda adalah asisten AI yang membantu menjelaskan hasil clustering berdasarkan data yang diberikan."),
                    ("human", f"Berikut adalah data clustering:\n{contexts}\n\nPertanyaan: {question}")
                ])
                messages = prompt.format_messages()
                response = llm.invoke(messages)
                st.write("Jawaban AI:")
                st.write(response.content)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses permintaan AI: {e}")
    else:
        st.info("Silakan unggah dataset Anda untuk memulai analisis.")