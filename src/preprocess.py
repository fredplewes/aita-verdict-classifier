import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp("Test text for spacy. Has some [punctuation!] and other stuff")

# Some reddit specific changes we have to make:
# It will be useful for us to keep tags like M23 and F31
# Unfortunately, spacy will treats M and F differently when lemmatising
# We need to substitute M and F with X and Y, since they get treated the same
# TODO: Fix M and F


# TODO: Decide whether or not to keep stopwords based on what gets better results
processed_text = " ".join([token.lemma_.lower() for token in doc if token.text.isalnum() and not token.is_stop])

print(f"Processed Text: {processed_text}")