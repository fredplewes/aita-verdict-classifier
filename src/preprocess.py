import spacy

nlp = spacy.load('en_core_web_sm')

text = "I, 20M, am trying to process this text. Running into some issues, but no matter."

doc = nlp(text)

# It will be useful for us to keep tags like M23, F16, 12M, 43F
# Spacy lemmatises 12M to 12 m so we need to merge manually
with doc.retokenize() as retokenizer:
    for i in range(len(doc)-1):
        if doc[i].text.isalnum() and doc[i+1].text.lower() == "m":
            retokenizer.merge(doc[i:i+2])

# TODO: Decide whether or not to keep stopwords based on what gets better results
processed_text = " ".join([token.lemma_.lower() for token in doc if token.text.isalnum() and not token.is_stop])

print(f"Processed Text: {processed_text}")