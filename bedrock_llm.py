import os
from google import genai

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

EMBED_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"


def embed_text(text: str):
    """
    Returns a list[float] embedding.
    """

    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text
    )

    return response.embeddings[0].values


def generate_answer(prompt: str):

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return response.text

if __name__ == "__main__":
    print(generate_answer("Hello"))
