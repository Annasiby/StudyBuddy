from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")   # a small, free "meaning machine"

facts = [
    "Velociraptor was about the size of a turkey and had feathers.",
    "Tyrannosaurus rex had a bite force strong enough to crush bones.",
    "Triceratops had three horns and a big bony frill on its head.",
    "Ornithomimus was one of the fastest dinosaurs, running about 50 mph.",
    "Brachiosaurus was a gentle giant that ate leaves from tall trees.",
    "Dinosaurs went extinct about 66 million years ago.",
]
card_vectors = model.encode(facts)                # one address per fact

def find_cards(question, top_k=2):
    q = model.encode([question])[0]
    scores = card_vectors @ q / (
        np.linalg.norm(card_vectors, axis=1) * np.linalg.norm(q)
    )                                             # how close is each card?
    best = np.argsort(scores)[::-1][:top_k]       # pick the closest ones
    return [facts[i] for i in best]

print(find_cards("Which one ate plants"))