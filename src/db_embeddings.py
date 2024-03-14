import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()


class DBEmbeddings:
    def __init__(self) -> None:
        self.connection = psycopg2.connect(
            database=os.getenv("DB_NAME"),
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT"),
        )
        self.cursor = self.connection.cursor()
        self.limit = 3
        self.similarity_threshold = 1

    def write_to_db(self, texts, embeddings):
        try:
            self.cursor.execute(
                "INSERT INTO embeddings (embedding, text) VALUES (%s, %s)",
                (embeddings, texts),
            )
            self.connection.commit()
        except (Exception, psycopg2.Error) as error:
            print("Error while writing to DB", error)

    def get_embeddings(self, text):
        message = "SELECT text, embedding <-> %s::vector AS similarity FROM embeddings ORDER BY similarity ASC LIMIT %s;"
        try:
            self.cursor.execute(message, (text, self.limit))
            results = self.cursor.fetchall()
            print(results)
            similarities = [x for x in results if x[1] < self.similarity_threshold]
            print(similarities)
            return similarities
        except (Exception, psycopg2.Error) as error:
            print("Error while getting embeddings from DB", error)
