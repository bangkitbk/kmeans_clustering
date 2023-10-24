from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
import random

# Fungsi untuk optimasi jumlah cluster
def optimize_cluster(X):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss)
    ax.set_title('Metode Elbow')
    ax.set_xlabel('Jumlah cluster')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)

# Fungsi untuk melakukan proses clustering
def perform_clustering(X, k, num_iters, c):
    # Inisialisasi nilai centroid awal
    centroids = c

    # Lakukan iterasi sebanyak num_iters
    for i in range(num_iters):
        # Hitung jarak tiap data point ke setiap centroid
        distances = cdist(X, centroids, 'euclidean')

        # Ambil index centroid dengan jarak terdekat
        labels = np.argmin(distances, axis=1)

        # Hitung nilai centroid baru
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # Cek apakah nilai centroid baru sudah konvergen atau belum
        if np.all(centroids == new_centroids):
            break

        # Perbarui nilai centroid
        centroids = new_centroids

        # Buat tabel untuk menampilkan nilai centroid setiap iterasi
        st.write(f"Iterasi ke-{i+1}:")
        centroid_table = pd.DataFrame(centroids, columns=[f'Feature {j+1}' for j in range(X.shape[1])])
        centroid_table['Cluster'] = np.arange(k)
        centroid_table.set_index('Cluster', inplace=True)
        st.write("Nilai centroid:")
        st.table(centroid_table)

        # Buat tabel untuk menampilkan hasil setiap iterasi
        st.write(f"Iterasi ke-{i+1}:")
        result_table = pd.DataFrame(X, columns=[f'Feature {j+1}' for j in range(X.shape[1])])
        result_table['Cluster'] = labels
        result_table['Euclidean'] = distances[np.arange(len(distances)), labels]
        st.write(result_table)
        
    # Menampilkan hasil clustering
    data_clustered = pd.DataFrame({'Data': st.session_state.df['Nama Barang'], 'Cluster': labels})
    st.write("Hasil Clustering")
    st.table(data_clustered)
    
    count_cluster = data_clustered.groupby('Cluster').count()
    plt.bar(count_cluster.index, count_cluster['Data'])
    plt.xlabel('Cluster')
    plt.ylabel('Jumlah Data')
    plt.title('Jumlah Data pada Setiap Cluster')
    st.pyplot(plt)
    
    fig, ax = plt.subplots()
    ax.scatter(X[labels == 0, 0], X[labels == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    ax.scatter(X[labels == 1, 0], X[labels == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    ax.scatter(X[labels == 2, 0], X[labels == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    ax.scatter(X[labels == 3, 0], X[labels == 3, 1], s = 100, c = 'Purple', label = 'Cluster 4')
    ax.scatter(X[labels == 4, 0], X[labels == 4, 1], s = 100, c = 'orange', label = 'Cluster 5')
    ax.scatter(X[labels == 5, 0], X[labels == 5, 1], s = 100, c = 'black', label = 'Cluster 6')
    ax.scatter(centroids[:, 0], centroids[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    ax.set_title('Clusters of data')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.legend()

    # Display the plot using st.pyplot()
    st.pyplot(fig)
    
    # hitung silhouette score
    silhouette_avg = silhouette_score(X, labels)

    # tampilkan tabel nilai pusat akhir dan silhouette coefficient
    centers_df = pd.DataFrame(data=centroids, columns=['X', 'Y'])
    st.write("Nilai Pusat Akhir:")
    st.write(centers_df)
    st.write("Nilai Silhouette Score :")
    st.write(silhouette_avg)
    
    # Menampilkan data untuk setiap cluster beserta nilai centroid tiap cluster
    for i in range(k):
        st.write(f"Data Barang Cluster Ke {i+1}:")
        st.write(f"Nilai Centroid : {np.round(centroids[i], decimals=0)}" )
        cluster_data = X[labels == i]
        cluster_table = pd.DataFrame(X[labels == i], columns=[f'Feature {j+1}' for j in range(X.shape[1])])
        cluster_table['Nama Barang'] = st.session_state.df['Nama Barang'][labels == i].reset_index(drop=True)
        st.write(f"Jumlah Data: {len(cluster_data)}")
        st.write(cluster_table)
    return(data_clustered, new_centroids)
        

# Main program
def main():
    st.sidebar.write("# Menu")
    menu_options = ["Input Data", "Optimasi Cluster", "Proses Clustering", "Proses Data Uji", "Cek Akurasi"]
    selected_menu = st.sidebar.selectbox("Pilih menu:", menu_options)

    if "df" not in st.session_state:
        st.session_state.df = None
    if "centroids" not in st.session_state:
        st.session_state.centroids = None

    if selected_menu == "Input Data":
        dataset = st.file_uploader("Upload a CSV")
        if dataset is not None:
            st.session_state.df = pd.read_csv(dataset)
            st.write(st.session_state.df)

    elif selected_menu == "Optimasi Cluster":
        if st.session_state.df is not None:
            X = st.session_state.df.iloc[:, [2, 3]].values
            optimize_cluster(X)
        else:
            st.write("Silakan upload data terlebih dahulu.")

    elif selected_menu == "Proses Clustering":
        if st.session_state.df is not None:
            X = st.session_state.df.iloc[:, [2, 3]].values
            k = st.selectbox('Pilih jumlah cluster:', list(range(1, 12)))
            k = int(k)
            num_iters = 100
            centroids = random.sample(X.tolist(), k)
            centroids = sorted(centroids)

            submit_button = st.button('Proses Clustering')
            if submit_button:
                st.session_state.df1,st.session_state.cf1 = perform_clustering(X, k, num_iters,centroids)
                
        else:
            st.write("Silakan upload data terlebih dahulu.")

    elif selected_menu == "Proses Data Uji":
        if st.session_state.df is not None:
            X = st.session_state.df.iloc[:, [4,5]].values
            k = st.selectbox('Pilih angka', list(range(1, 12)))
            num_iters = 100
            centroids = st.session_state.cf1
            submit_button = st.button('Proses Data Uji')
            if submit_button:
                st.session_state.df2,st.session_state.cf2 = perform_clustering(X, k, num_iters, centroids)
                

            
        else:
            st.write("Silakan upload data dan lakukan proses clustering terlebih dahulu.")

    elif selected_menu == "Cek Akurasi":
        # st.write(st.session_state.df1)
        # st.write(st.session_state.df2)
        df_temp = st.session_state.df1.copy()
        df_temp2 = st.session_state.df2.copy()
        # df_temp.set_index('Data', inplace = True)
        df_temp['Cluster 2'] = df_temp2['Cluster']
        st.write(df_temp)
        # Mengambil kolom Cluster dari dataframe pertama dan kedua
        labels_true = st.session_state.df1['Cluster'].values
        labels_pred = st.session_state.df2['Cluster'].values

        # Menghitung akurasi
        accuracy = accuracy_score(labels_true, labels_pred)

        # Menampilkan hasil akurasi
        st.write(f"Akurasi: {accuracy}")
        ##st.write(st.session_state.cf2)
        
# Panggil fungsi main
if __name__ == '__main__':
    main()
