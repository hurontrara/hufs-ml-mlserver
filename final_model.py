# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # K-Means
from sklearn.metrics import silhouette_score, silhouette_samples  # 실루엣 계수 계산

# %% md
### data 읽어오기
# %%
data = pd.read_csv("final_data")
data_size = len(data)
label = data['label']
# %% md
### 단어에 word2vec 적용하기
# %%
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import gensim
import numpy as np

# Load a pre-trained Word2Vec model (replace 'path/to/your/model' with the actual path)
model = gensim.models.Word2Vec.load('ko.bin')
print(model.wv.vectors.shape)  # 모델 내 단어 갯수는 30185개

# List of words to be labeled
word_list = label

# Create a dictionary to store word vectors
word_vectors = {}
NaN = []

# Populate the dictionary with word vectors
for word in word_list:
    try:
        new_word = word.strip(" ")
        word_vectors[new_word] = model[new_word]
    except KeyError:
        # Handle the case where the word is not in the vocabulary
        print(f"Word '{new_word}' not in vocabulary.")
        NaN.append(word)
word_vectors
word_df = pd.DataFrame(word_vectors)
word_df
# %%
print(list(set(NaN)))
# %%
word_matrix = word_df.transpose()
word_matrix
# %% md
### 3차원 시각화
# %%
import numpy as np
from sklearn.decomposition import PCA

# Number of desired dimensions after PCA
n_components = 3

# Standardize the features (optional but recommended for PCA)
data_standardized = (word_matrix - np.mean(word_matrix, axis=0)) / np.std(word_matrix, axis=0)

# Initialize PCA with the desired number of components
pca = PCA(n_components=n_components)

# Fit PCA and transform the data
data_reduced = pca.fit_transform(data_standardized)

# The variable data_reduced now contains the data in 24 dimensions

# Print the explained variance ratio for each selected component
print("Explained Variance Ratio for each component:")
print(pca.explained_variance_ratio_)

# Optionally, if you want to access the principal components themselves
principal_components = pca.components_

# You can also access the cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Print the cumulative explained variance
print("\nCumulative Explained Variance:")
print(cumulative_explained_variance)
# %%
data_reduced
# %% md
##### 최적의 cluster 찾기
# %%
distortions = []

for i in range(2, 100):
    kmeans_i = KMeans(n_clusters=i, random_state=0)  # 모형 생성
    kmeans_i.fit(data_reduced)  # 모형 훈련
    distortions.append(kmeans_i.inertia_)

plt.plot(range(2, 100), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
# %%
silhouette = []

for i in range(2, 100):
    kmeans_i = KMeans(n_clusters=i, random_state=0)  # 모형 생성
    kmeans_i.fit(data_reduced)  # 모형 훈련
    silhouette_values = silhouette_score(data_reduced, kmeans_i.labels_)
    silhouette.append(silhouette_values)

plt.plot(range(2, 100), silhouette, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Shilhouette coefficient')
plt.show()
# %%
from matplotlib import cm


def silhouetteViz(n_cluster, X_features):
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    Y_labels = kmeans.fit_predict(X_features)

    silhouette_values = silhouette_samples(X_features, Y_labels, metric='euclidean')

    y_ax_lower, y_ax_upper = 0, 0
    y_ticks = []

    for c in range(n_cluster):
        c_silhouettes = silhouette_values[Y_labels == c]
        c_silhouettes.sort()
        y_ax_upper += len(c_silhouettes)
        color = cm.jet(float(c) / n_cluster)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouettes,
                 height=1.0, edgecolor='none', color=color)
        y_ticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouettes)

    silhouette_avg = np.mean(silhouette_values)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.title('Number of Cluster : ' + str(n_cluster) + '\n' \
              + 'Silhouette Score : ' + str(round(silhouette_avg, 3)))
    plt.yticks(y_ticks, range(n_cluster))
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()
    plt.show()


# %%
silhouetteViz(18, data_reduced)

# %%
kmeans = KMeans(n_clusters=18, random_state=42)
res1 = pd.DataFrame(data_reduced)
model = kmeans.fit_predict(data_reduced)
res1["Cluster"] = model
res1
# %%
res2 = res1.copy()
res2.index = word_matrix.index
res2
# %%
res2[res2['Cluster'] == 1]
# %%

df = res2

# Visualize the 3D clusters
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for cluster in range(df['Cluster'].nunique()):
    cluster_data = df[df['Cluster'] == cluster]
    ax.scatter(cluster_data[0], cluster_data[1], cluster_data[2], label=f'Cluster {cluster}')

ax.set_xlabel('Feature 0')
ax.set_ylabel('Feature 1')
ax.set_zlabel('Feature 2')
ax.set_title('3D Cluster Visualization')
ax.legend()

plt.show()
# %%
!pip
install
plotly

# %%
import plotly.graph_objects as go

# Assuming your DataFrame is named 'res3'

# Create a 3D scatter plot using Plotly
fig = go.Figure()

for cluster in range(res2['Cluster'].nunique()):
    cluster_data = res2[res2['Cluster'] == cluster]
    fig.add_trace(go.Scatter3d(
        x=cluster_data[0],
        y=cluster_data[1],
        z=cluster_data[2],
        mode='markers',
        marker=dict(size=5, color=cluster, opacity=0.7),
        name=f'Cluster {cluster}'
    ))

# Layout settings
fig.update_layout(
    scene=dict(
        xaxis_title='Feature 0',
        yaxis_title='Feature 1',
        zaxis_title='Feature 2',
        aspectmode='cube'
    ),
    title='Interactive 3D Cluster Visualization',
    width=800,  # Adjust the width as needed
    height=600  # Adjust the height as needed
)

# Show the interactive plot in the Jupyter Notebook
fig.show()

# %% md
## 최적의 성능을 내는 모델
# %% md
#### 적절한 PCA 값 찾기 1
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load a sample dataset (replace this with your own data)

# Standardize the features (optional but recommended for PCA)
X_standardized = (word_matrix - np.mean(word_matrix, axis=0)) / np.std(word_matrix, axis=0)

# Fit PCA and obtain the eigenvalues
pca = PCA()
pca.fit(X_standardized)
eigenvalues = pca.explained_variance_

# Plot the scree plot
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.title('Scree Plot')
plt.xlabel('Number of Principal Components')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.show()
# %% md
##### 5~ 15차원 사이의 값이 PCA에 가장 적절하다
# %% md
#### 적절한 PCA 값 찾기 2

# %%
! pip
install
optuna
# %%
import optuna
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline

X = word_matrix


# Define an objective function to optimize
def objective(trial):
    # Define the search space for hyperparameters
    n_components = trial.suggest_int('n_components', 3, 30)  # Number of components in PCA
    n_clusters = trial.suggest_int('n_clusters', 10, 30)  # Number of clusters in KMeans

    # Create a pipeline with PCA and KMeans
    pipeline = Pipeline([
        ('reduce_dim', PCA(n_components=n_components)),
        ('cluster', KMeans(n_clusters=n_clusters))
    ])

    # Fit the pipeline
    pipeline.fit(X)

    # Evaluate the performance using silhouette score
    silhouette_avg = silhouette_score(X, pipeline.named_steps['cluster'].labels_)
    return silhouette_avg


# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')  # maximize silhouette score
study.optimize(objective, n_trials=50)  # perform 50 trials

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# %% md

# %% md
#### 적절한 PCA 값 찾기 3

# %%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Fit PCA
pca = PCA()
pca.fit(word_matrix)

# Plot explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.show()

# %% md
### PCA로 차원 축소
# %%
import numpy as np
from sklearn.decomposition import PCA

# Number of desired dimensions after PCA
n_components = 7

# Standardize the features (optional but recommended for PCA)
data_standardized = (word_matrix - np.mean(word_matrix, axis=0)) / np.std(word_matrix, axis=0)

# Initialize PCA with the desired number of components
pca = PCA(n_components=n_components)

# Fit PCA and transform the data
data_reduced = pca.fit_transform(data_standardized)

# The variable data_reduced now contains the data in 24 dimensions

# Print the explained variance ratio for each selected component
print("Explained Variance Ratio for each component:")
print(pca.explained_variance_ratio_)

# Optionally, if you want to access the principal components themselves
principal_components = pca.components_

# You can also access the cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Print the cumulative explained variance
print("\nCumulative Explained Variance:")
print(cumulative_explained_variance)
# %% md
#### 적절한 cluster 값 찾기 1

# %%
distortions = []

for i in range(2, 100):
    kmeans_i = KMeans(n_clusters=i, random_state=0)  # 모형 생성
    kmeans_i.fit(data_reduced)  # 모형 훈련
    distortions.append(kmeans_i.inertia_)

plt.plot(range(2, 100), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
# %% md
#### 적절한 cluster 값 찾기 2

# %%
silhouette = []

for i in range(10, 100):
    kmeans_i = KMeans(n_clusters=i, random_state=0)  # 모형 생성
    kmeans_i.fit(data_reduced)  # 모형 훈련
    silhouette_values = silhouette_score(data_reduced, kmeans_i.labels_)
    silhouette.append(silhouette_values)

plt.plot(range(10, 100), silhouette, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Shilhouette coefficient')
plt.show()

# %% md
#### 적절한 cluster 값 찾기 3

# %%
from matplotlib import cm


def silhouetteViz(n_cluster, X_features):
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    Y_labels = kmeans.fit_predict(X_features)

    silhouette_values = silhouette_samples(X_features, Y_labels, metric='euclidean')

    y_ax_lower, y_ax_upper = 0, 0
    y_ticks = []

    for c in range(n_cluster):
        c_silhouettes = silhouette_values[Y_labels == c]
        c_silhouettes.sort()
        y_ax_upper += len(c_silhouettes)
        color = cm.jet(float(c) / n_cluster)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouettes,
                 height=1.0, edgecolor='none', color=color)
        y_ticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouettes)

    silhouette_avg = np.mean(silhouette_values)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.title('Number of Cluster : ' + str(n_cluster) + '\n' \
              + 'Silhouette Score : ' + str(round(silhouette_avg, 3)))
    plt.yticks(y_ticks, range(n_cluster))
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()
    plt.show()


# %%
silhouetteViz(18, data_reduced)

# %% md
### 결정된 PCA 차원, cluster 값으로 Kmeans 진행
# %%
kmeans = KMeans(n_clusters=18, random_state=42)
res1 = pd.DataFrame(data_reduced)
model = kmeans.fit_predict(data_reduced)
res1["Cluster"] = model
res1

# %%
res2 = res1.copy()
res2.index = word_matrix.index
res2
# %%
res2[res2['Cluster'] == 1]
# %%
