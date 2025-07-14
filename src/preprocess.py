import spacy
import pandas as pd
import os
from tqdm import tqdm

os.makedirs(os.path.dirname("data/processed/aita_processed.csv"), exist_ok=True)

nlp = spacy.load('en_core_web_sm')

def cleantext (text):
    doc = nlp(text)

    # It will be useful for us to keep tags like M23, F16, 12M, 43F
    # Spacy lemmatises 12M to 12 m so we need to merge manually
    with doc.retokenize() as retokenizer:
        for i in range(len(doc)-1):
            if doc[i].text.isalnum() and doc[i+1].text.lower() == "m":
                retokenizer.merge(doc[i:i+2])

    # TODO: Decide whether or not to keep stopwords based on what gets better results
    text = " ".join([token.lemma_.lower() for token in doc if token.text.isalnum() and not token.is_stop])
    return text

# TODO: This is incredibly slow, find something faster
df = pd.read_csv("data/aita_clean.csv")
# drop rows with missing title or body
df = df.dropna(subset=["title", "body"])
tqdm.pandas(desc="Cleaning post titles")
df["title"] = df["title"].progress_apply(cleantext)
tqdm.pandas(desc="Cleaning post contents")
df["body"] = df["body"].progress_apply(cleantext)
# drop empty rows again incase cleaning removed all text
# keeping in mind these are empty strings now instead of NaN
df = df[df["title"].str.strip().astype(bool) & df["body"].str.strip().astype(bool)]
df[["title", "body", "verdict"]].to_csv("data/processed/aita_processed.csv", index=False)
print("Preprocessing complete. Processed data saved to data/processed/aita_processed.csv")