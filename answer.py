from dotenv import load_dotenv
import anthropic
from sentence_transformers import SentenceTransformer, util
from main import load_chunks

load_dotenv()

USE_MOCK = True
THRESHOLD = 0.54

client = None if USE_MOCK else anthropic.Anthropic()

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chunks = load_chunks()
chunk_embs = model.encode([c["text"] for c in chunks])

SYSTEM = (
    "You are a customer support assistant for a clothing store. "
    "Answer ONLY from the reference texts you are given. "
    "If the answer is not in them, reply exactly: نمی دونم. "
    "Reply in  the same language as the question, in one or two short sentences. "
)


def retrieve(question, k=3):
    q_emb = model.encode(question)
    scores = util.cos_sim(q_emb, chunk_embs)[0].tolist()
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return ranked[:k]


def answer(question):
    top = retrieve(question)
    
    if USE_MOCK:
        best_score, best_chunk = top[0]
        if best_score < THRESHOLD:
            return "نمی دونم", top
        return best_chunk["text"], top
    
    context = "\n".join(f"[{c['source']}] {c['text']}" for _, c in top)
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=300,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Reference texts:\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return response.content[0].text, top


if __name__ == "__main__":
    for q in [
        "ارسال به تهران چند روز طول می کشه؟",
        "ساعت کاری فروشگاه چنده؟",
    ]:
        text, top = answer(q)
        print(text)
        print("منابع:", [c["source"] for _, c in top])
        print()