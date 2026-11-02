from sentence_transformers import SentenceTransformer, util
from main import load_chunks

chunks = load_chunks()
texts = [c["text"] for c in chunks]

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chunk_embs = model.encode(texts)

question = "ارسال به تهران چند روز طول می کشه؟"
q_emb = model.encode(question)

scores = util.cos_sim(q_emb, chunk_embs)[0].tolist()
ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

for score, chunk in ranked[:3]:
    print(f"{score:.2f} | {chunk['source']} | {chunk['text']}")