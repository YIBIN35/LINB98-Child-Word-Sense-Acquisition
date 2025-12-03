import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def generate_pca_graph(
        target_word,
        df,
        gloss_dict,
        embedder,
        save_path=None,
        title=None,
):

    # sort out sentences
    df_word = df[df["target"] == target_word].copy()
    df_word = df_word.sort_values(by="sense_label")

    unique_labels = df_word["sense_label"].unique()

    # Group the clusters
    clusters = []
    cluster_sizes = []
    for lab in unique_labels:
        sentences = df_word[df_word["sense_label"] == lab]["sentences"].tolist()
        clusters.append(sentences)
        cluster_sizes.append(len(sentences))

    # Create embeddings
    all_sentence_embeddings = []
    centroid_embeddings = []

    for sentences in clusters:
        cluster_vecs = [embedder.create_vectors_masked(s, target_word) for s in sentences]
        all_sentence_embeddings.extend(cluster_vecs)
        centroid_embeddings.append(np.vstack(cluster_vecs).mean(axis=0))

    gloss_embeddings = []
    for gloss in gloss_dict.values():
        gloss_embeddings.append(embedder.create_vectors_masked(gloss, target_word))

    combined = np.vstack(all_sentence_embeddings + centroid_embeddings + gloss_embeddings)

    # Apply PCA (2D space)
    pca = PCA(n_components=2, random_state=236)
    pca_vectors = pca.fit_transform(combined)

    # Organize sections for color-coding
    idx = 0
    cluster_points = []
    for size in cluster_sizes:
        cluster_points.append(pca_vectors[idx : idx + size])
        idx += size
    centroid_points = pca_vectors[idx : idx + len(centroid_embeddings)]
    gloss_points = pca_vectors[idx + len(centroid_embeddings) :]

    # Generate figure
    colors = ["tab:blue", "tab:purple", "tab:green", "tab:brown"]
    centroid_color = "tab:red"
    gloss_color = "tab:orange"
    plt.figure(figsize=(8,6))
    for pts, color, label in zip(cluster_points, colors, [f"cluster_{i+1}" for i in range(len(cluster_points))]):
        plt.scatter(pts[:,0], pts[:,1], color=color, label=label, alpha=0.9)
    plt.scatter(centroid_points[:,0], centroid_points[:,1], color=centroid_color, label="centroids", s=70)
    plt.scatter(gloss_points[:,0], gloss_points[:,1], color=gloss_color, label="glosses", s=70)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title or f"PCA Scatterplot ({target_word}, masked)")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved PCA graph for '{target_word}' to {save_path}")
