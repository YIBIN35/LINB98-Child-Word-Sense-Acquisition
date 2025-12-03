import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def generate_heatmaps(
        target_word,
        df,
        gloss_dict,
        embedder,
        save_path=None,
        top_n=None,
):
    # Filter words
    df_word = df[df["target"] == target_word].copy()
    df_word_grouped = df_word.sort_values(by=["sense_label", "distance_to_sense_mean"])

    unique_labels = df_word_grouped["sense_label"].unique()

    # Pick the top_n sentences to compute the average
    if top_n is not None:
        df_sorted = df_word_grouped.groupby("sense_label").head(top_n)

    # All sentence included
    else:
        df_sorted = df_word_grouped

    # Generate the embeddings
    clusters = {str(label): [] for label in unique_labels}

    for label in unique_labels:
        sentence_list = df_sorted.loc[df_sorted["sense_label"] == label, "sentences"].tolist()
        clusters[str(label)] = sentence_list

    # masked vectors (unmasked vectors not included for now)

    # clusters
    masked_cluster_avg = []
    for label, sentences in clusters.items():
        masked_lst = []
        for sentence in sentences:
            masked_vec = embedder.create_vectors_masked(sentence, target_word)
            masked_lst.append(masked_vec)

        masked_avg = np.vstack(masked_lst).mean(axis=0)
        masked_cluster_avg.append(masked_avg)

    # glosses
    masked_gloss_avg = []
    for gloss in gloss_dict.values():
        gloss_vec_masked = embedder.create_vectors_masked(gloss, target_word)
        masked_gloss_avg.append(gloss_vec_masked)




    # Compute cosine similarity
    cosine_masked = cosine_similarity(masked_cluster_avg, masked_gloss_avg)
    # cosine_unmasked = cosine_similarity(unmasked_cluster_avg, unmasked_gloss_avg)

    # Generate graph (only masked for now)
    num_clusters = len(masked_cluster_avg)
    cluster_names = [f"cluster_{i+1}" for i in range(num_clusters)]
    gloss_names = list(gloss_dict.keys())

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        cosine_masked,
        xticklabels=gloss_names,
        yticklabels=cluster_names,
        cmap="coolwarm",
        vmin=0.5,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Cosine similarity"},
    )
    plt.xlabel("Glosses")
    plt.ylabel("Clusters")
    plt.title(f"Cosine Similarity Heatmap (Masked, {target_word})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved PCA graph for '{target_word}' to {save_path}")

