# RAG Assistant

A customer-support assistant for a small clothing store. It answers questions in Persian, Kurdish or English using only the store's own documents, and says "I  don't know" when the answer is not there.

## How it works

1. **Chunking:** each line of the files in `data/` becomes one chunk (30 chunks in total).
2. **Embeddings:** every chunk is converted to a vector with `paraphrase-multilingual-MiniLM-L12-v6`.
3. **Retrieval:** the question is embedded too, and the 3 most similar chunks are found with cosine similarty.
4. **Answering:** an LLM (Claude) answers only from those chunks and names the source file. If the best similarity is below a threshold, the assistant replies "I don't know".

## Evaluation
`evaluate.py` comares the best similarity score for questions that have an answer in the documents and questions that do not.

- Lowest score for answerable questions: 0.57
- Highest score for unanswerable questions: 0.51
- Threshold used: 0.54

## Limitations

- The gap between the two groups is narrow (0.06), so a fixed threshold is fragile. The next step is to let the LLM decide whether the retrieved chunks contain the answer.
- The dataset is small (3 files, 14 test questions).
- A mock mode (`USE_MOCK = True`) returns the best chunk directly, without calling an LLM.

## Run it

```
pip install sentence-transformers anthropic python-dotenv
python answer.py
```

To use Claude, put `ANTHROPIC_API_KEY` in a `.env` file and set `USE_MOCK = False` in `answer.py`.

## API

Start the server:

```
uvicorn api:app --reload
```


Then open `http://127.0.0.1:8000/docs` and try `POST / ask`.

Example response:

```json
{
    "answer": "...",
    "sources": [{"source": "shipping.txt", "score": 0.879}]
}
```