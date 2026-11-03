from sentence_transformers import SentenceTransformer, util
from main import load_chunks

model = SentenceTransformer("paraphrase-multilingual-MinilM-L12-v2")
chunks = load_chunks()
chunk_embs = model.encode([c["text"] for c in chunks])


def best_score(question):
    q_emb = model.encode(question)
    scores = util.cos_sim(q_emb, chunk_embs) [0]
    return float(scores.max())


with open("questions.txt", encoding="utf-8") as f:
    answerable = [line.strip() for line in f if line.strip()]
    
    
unanswerable = [
    "ادرس فروشگاه کجاست؟",
    "ساعت کاری فروشگاه چنده؟",
    "ایا دوربین هم می فروشد؟",
    "شماره تلفن پشتیبانی چنده؟",
    "ایا کیف هم دارید؟",
]

print("--- سوال های دارای جواب ---")
for q in answerable:
    print(f"{best_score(q):.2f} | {q}")
    
print("--- سوال های بی جواب ---")
for q in unanswerable:
    print(f"{best_score(q):.2f} | {q}")
    
    
answer_scores = [best_score(q) for q in answerable]
no_answer_scores = [best_score(q) for q in unanswerable]
print("کمترین عدد دارای جواب:", round(min(answer_scores), 2))
print("بیشترین عدد بی جواب:", round(max(no_answer_scores), 2))