# Load the saved model and predict

import spacy
output_dir = '../Product-Keyword-Extraction-with-NLP'

print("Loading from", output_dir)
nlp_updated = spacy.load(output_dir)
doc = nlp_updated("Fridge can be ordered in FlipKart" )
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])