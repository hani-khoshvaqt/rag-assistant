import os


def load_chunks():
    chunks = []
    for name in os.listdir("data"):
        path = os.path.join("data", name)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for line in text.splitlines():
            if line.strip():
                chunks.append({"source": name, "text": line.strip()})
    return chunks


if __name__ == "__main__":
    chunks = load_chunks()
    print(len(chunks))
    print(chunks[0])