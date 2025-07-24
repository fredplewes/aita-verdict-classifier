import spacy
import pandas as pd
import os
from tqdm import tqdm

os.makedirs(os.path.dirname("data/processed/aita_processed.csv"), exist_ok=True)

nlp = spacy.load('en_core_web_sm')

def cleandoc(doc):
    # It will be useful for us to keep tags like M23, F16, 12M, 43F
    # Spacy lemmatises 12M to 12 m so we need to merge manually
    # TODO: Decide whether to keep these number tags or not
    with doc.retokenize() as retokenizer:
        for i in range(len(doc)-1):
            if doc[i].text.isalnum() and doc[i+1].text.lower() == ("m" or "f"):
                retokenizer.merge(doc[i:i+2])

    # TODO: Decide whether or not to keep stopwords based on what gets better results
    text = " ".join([token.lemma_.lower() for token in doc if token.text.isalnum() and not token.is_stop])
    return text

if __name__ == "__main__":
    df = pd.read_csv("data/aita_clean.csv")
    # drop rows with missing title or body
    df = df.dropna(subset=["title", "body"])

    SAMPLE_SIZE = 3000
    SEED = 123
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)

    print(f"Processing {len(df)} posts, this may take a while...")
    tqdm.pandas(desc="Cleaning post titles")
    title_docs = list(tqdm(nlp.pipe(df["title"], batch_size=100), total=len(df["title"]), desc="Cleaning post titles"))
    df["title"] = [cleandoc(doc) for doc in title_docs]

    tqdm.pandas(desc="Cleaning post contents")
    body_docs = list(tqdm(nlp.pipe(df["body"], batch_size=100), total=len(df["title"]), desc="Cleaning post contents"))
    df["body"] = [cleandoc(doc) for doc in body_docs]

    # drop empty rows again incase cleaning removed all text
    # keeping in mind these are empty strings now instead of NaN
    df = df[df["title"].str.strip().astype(bool) & df["body"].str.strip().astype(bool)]
    df[["title", "body", "verdict"]].to_csv("data/processed/aita_processed.csv", index=False)
    print("Preprocessing complete. Processed data saved to data/processed/aita_processed.csv")