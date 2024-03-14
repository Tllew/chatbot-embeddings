from src.gpt_adapter import OpenAIAdapter
from src.embeddings import make_embeddings

if __name__ == "__main__":
    make_embeddings()
    oi = OpenAIAdapter()
    oi.chat()