import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from openai import OpenAI

# df = pd.read_csv('sentence_data_bert_dpgmm100k_wordsetlarge_mixedSentFalse_pca2d_priorNone.csv')

# Load GPT with own API
client = OpenAI(api_key="API key here")

def label_with_llm(df,
                   target_word,
                   definitions,
                   sentence_col = "sentences",
                   model = "gpt-4o-mini",
                   top_n = None,
                   save_path_csv = None,
                   save_path_histogram = None):

    # Process the given dataframe
    df_word = df[df["target"] == target_word].copy()

    # Pick the top_n sentences to compute the average
    if top_n is not None:
        df_sorted = df_word.groupby("sense_label").head(top_n)

    else:
        df_sorted = df_word

    # All sentence included
    # else:
    #     df_sorted = df_word.groupby("sense_label")
    #     unique_labels = df_sorted["sense_label"].unique()
    #     cluster_lens = []
    #     for label in unique_labels:


    results = []
    def_processed = "\n".join(f"{key}: {value}" for key, value in definitions.items())

    for idx, row in df_sorted.iterrows():
        sentence = row[sentence_col]

        prompt = (
            "You are a word definition classifier. Based on the sentences and available definitions, "
            "choose the best-fitting definition and assign a confidence score.\n\n"
            f"Sentence index: {idx}\n"
            f"Sentence: {sentence}\n\n"
            f"Definitions:\n{def_processed}\n\n"
            "Return the result as a JSON object with EXACTLY the following structure:\n"
            "{\n"
            '  "sentence": "<the full sentence>",\n'
            '  "definition_key": "<the chosen definition key>",\n'
            '  "definition": "<the chosen definition text>",\n'
            '  "confidence": <number between 0 and 1>\n'
            "}\n\n"
            "Return ONLY valid JSON. "
            "Do NOT wrap your answer in a code block. "
            "Do NOT include ```json or any backticks. "
            "Your response must start with '{' and end with '}'. "
            "Any extra text makes the answer invalid."
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        content = resp.choices[0].message.content.strip()
        print("RAW OUTPUT:\n", content)
        parsed = json.loads(content)
        results.append(parsed)

    results_df = pd.DataFrame(results)

    if save_path_csv:
        results_df.to_csv(save_path_csv, index=False)
        print(f"Saved results to: {save_path_csv}")

    # Generate the histogram
    histogram_df = results_df.copy()

    # Assign cluster IDs based on row order
    cluster_size = top_n
    histogram_df["cluster_number"] = histogram_df.index // cluster_size + 1

    # Generate the frequency histogram
    for num, group in histogram_df.groupby("cluster_number"):
        counts = group["definition_key"].value_counts()

        plt.figure(figsize=(6, 4))
        plt.bar_width = 0.3
        plt.ylim(0, 55)
        plt.gca().yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        plt.bar(counts.index, counts.values)
        for i, v in enumerate(counts.values):
            plt.text(i, v + 0.1, str(v), ha='center', va='bottom')
        plt.xlabel("Definition Key")
        plt.ylabel("Frequency")
        plt.title(f"Cluster {num} — Definition Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path_histogram:
            plt.savefig(save_path_histogram, dpi=300)
            plt.close()
        print(f"Saved frequency histogram, cluster {num} for '{target_word}' to {save_path_histogram}")

    # Generate confidence score histogram
    for num, group in histogram_df.groupby("cluster_number"):
        plt.figure(figsize=(6, 4))

        # histogram: bins from 0.0 to 1.0
        bins = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
        plt.hist(group["confidence"], bins=bins, edgecolor="black", color="skyblue")

        plt.ylim(0, 55)

        plt.xlabel("Confidence Score")
        plt.ylabel("Frequency")
        plt.title(f"Cluster {num} — Confidence Histogram")

        plt.gca().yaxis.set_major_locator(mtick.MaxNLocator(integer=True))

        plt.tight_layout()

        if save_path_histogram:
            plt.savefig(save_path_histogram, dpi=300)
            plt.close()
        print(f"Saved confidence score, cluster {num} for '{target_word}' to {save_path_histogram}")
