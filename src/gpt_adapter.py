from openai import OpenAI
from dotenv import load_dotenv
from lib.db_embeddings import DBEmbeddings
from lib.embeddings import get_embedding

load_dotenv()

class OpenAIAdapter:
    dbembeddings = DBEmbeddings()

    def __init__(self):
        self.client = OpenAI()

    def get_extra_context(self):
        return "From incredible animation to a super story and tons of Easter eggs, Spider-Man: Across the Spider-Verse has everything fans could ask for."

    def get_db_context(self, message):
        message_embedding = get_embedding(message)
        embeddings = self.dbembeddings.get_embeddings(message_embedding)
        return embeddings

    def get_context(self):
        return "You are a film critic. Only answer based on the context provided below, if you can't say 'I don't know'."

    def parse_db_results(self, db_results):
        return "\n".join([x[0] for x in db_results])

    def get_response(self, message):
        db_results = self.get_db_context(message)
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # {"role": "system", "content": self.get_context() + self.get_extra_context()},
                # {"role": "system", "content": self.get_context() + self.parse_db_results(db_results) },
                {"role": "user", "content": message},
            ]
        )
        return response.choices[0].message.content
    
    def chat(self):
        while True:
            message = input("You: ")
            response = self.get_response(message)
            print(f"Custom Assistant: {response}")