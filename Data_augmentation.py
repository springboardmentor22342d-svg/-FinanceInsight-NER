import random
import nltk
from nltk.corpus import wordnet
import spacy

nltk.download('wordnet')
nltk.download('omw-1.4')

# Load SpaCy NER model
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------
# 1️⃣ SYNONYM REPLACEMENT
# ---------------------------------------------------------
def synonym_replacement(text, n=2):
    words = text.split()
    new_words = words.copy()
    
    random_words = list(set([w for w in words if w.isalpha()]))
    random.shuffle(random_words)
    
    replaced = 0
    for word in random_words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            syn_list = set()
            for s in synonyms:
                for lemma in s.lemmas():
                    syn_list.add(lemma.name().replace("_", " "))
            
            syn_list.discard(word)
            if len(syn_list) > 0:
                new_word = random.choice(list(syn_list))
                new_words = [new_word if w == word else w for w in new_words]
                replaced += 1
        
        if replaced >= n:
            break
    
    return " ".join(new_words)

# ---------------------------------------------------------
# 2️⃣ ENTITY MASKING
# ---------------------------------------------------------
def entity_masking(text):
    doc = nlp(text)
    masked_text = text
    
    for ent in doc.ents:
        masked_text = masked_text.replace(ent.text, f"<{ent.label_}>")
    
    return masked_text

# ---------------------------------------------------------
# 3️⃣ OFFLINE BACK-TRANSLATION (simple reverse logic)
# ---------------------------------------------------------
def back_translate(text):
    # Offline fallback since no deep-translator
    words = text.split()
    reversed_words = words[::-1]
    return " ".join(reversed_words)

# ---------------------------------------------------------
# 4️⃣ APPLY ALL AUGMENTATION METHODS
# ---------------------------------------------------------
def augment_text(text):
    return {
        "original": text,
        "synonym_replacement": synonym_replacement(text),
        "entity_masking": entity_masking(text),
        "back_translation": back_translate(text)
    }

# ---------------------------------------------------------
# 5️⃣ TEST
# ---------------------------------------------------------
sample = "The rupee strengthened against the US dollar in early trade."

augmented = augment_text(sample)

for k, v in augmented.items():
    print(f"\n--- {k.upper()} ---")
    print(v)
