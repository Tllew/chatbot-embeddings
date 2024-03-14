from lib.gpt_adapter import OpenAIAdapter
from lib.embeddings import make_embeddings

if __name__ == "__main__":
    # make_embeddings()
    oi = OpenAIAdapter()
    oi.chat()