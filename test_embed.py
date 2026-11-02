from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

a = "ارسال به تهران چند روز طول می کشه؟"
b = "ارسال به تهران 2 تا 3 روز کاری طول می کشه."
c = "قیمت کت کوتاه 2 میلیون و 300 هزار تومانه."

emb = model.encode([a, b, c])
print(util.cos_sim(emb[0], emb[1]))
print(util.cos_sim(emb[0], emb[2]))