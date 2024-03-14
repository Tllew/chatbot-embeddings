from openai import OpenAI
import os
from dotenv import load_dotenv
from src.db_embeddings import DBEmbeddings

load_dotenv()
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_text_embeddings(text):
    # split the text into lines and get the embeddings
    text = text.split("\n")
    results = {}
    for line in text:
        results[line] = get_embedding(line)
    return results

def save_file(filename, embeddings):
    # write csv with headings "text" and "embedding"
    with open(filename, "w") as file:
        file.write("text,embedding\n")
        for text, embedding in embeddings.items():
            file.write(f"{text},{embedding}\n")

def load_source_file(filename):
    # load a file and get the embeddings
    with open(filename, "r") as file:
        embeddings = get_text_embeddings(file.read())
        return embeddings

def make_embeddings():
    # load all .txt files in the embeddings directory
    dbembedding = DBEmbeddings()

    for filename in os.listdir("embeddings"):
        if filename.endswith(".txt"):
            name = filename.split('.txt')[0]
            csv_filename = f'embeddings/{name}.csv'
            
            embeddings = load_source_file(f'embeddings/{filename}')
            save_file(csv_filename, embeddings)
            for text, embedding in embeddings.items():
                dbembedding.write_to_db(text, embedding)