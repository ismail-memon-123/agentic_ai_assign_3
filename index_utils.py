import numpy as np
import json
from bedrock_llm import embed_text, generate_answer


def normalize(vectors):
    """
    Normalize vectors for cosine similarity
        """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def initial(embedding_file="index/vectors.npy", chunks_file="index/metadata.json"):
    """
    Load embeddings and document chunks
    """
    embeddings = np.load(embedding_file)

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Normalize embeddings once
    embeddings = normalize(embeddings)
    print("embedding len")
    print(str(len(embeddings)))
    return chunks, embeddings

def search(chunks, embeddings, query_embedding, top_k=5):
    """
        Cosine similarity search
        """

    # normalize query vector
    query_embedding = query_embedding/np.linalg.norm(query_embedding)


    # cosine similarity
    scores = np.dot(
        embeddings,
        query_embedding
    )


        # highest scores first
    print(top_k)
    best_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    print(best_indices)
    for idx in range(len(best_indices)):
        results.append({
            "score": float(scores[idx]),
            "text": chunks[idx]["text"],
            "id": idx
        })


    return results

def retrieve(question, top_k=5):

    # create query embedding
    query_vector = embed_text(
        question
    )

    chunks, embeddings = initial()
    results = search(
        chunks, embeddings, query_vector,
        top_k
    )


    return results

def build_prompt(question, results):

    """
    Create grounded RAG prompt
    """

    context = ""

    for i, result in enumerate(results):

        citation = f"[Source {i+1}]"


        context += f"""
{citation}
{result['text']}

"""

    prompt = f"""

You are a helpful assistant.

Answer the user's question ONLY using the provided context.

If the answer is not contained in the context,
say "I don't have enough information."

Always cite sources using [Source X] format.


Context:

{context}


Question:

{question}


Answer:

"""


    return prompt

def main():

    question = input(
        "\nQuestion: "
    )


    # 1. Retrieve documents

    results = retrieve(
        question
    )


    # 2. Build grounded prompt

    rag_prompt = build_prompt(
        question,
        results
    )


    # 3. Call LLM

    answer = generate_answer(
        rag_prompt
    )


    # 4. Print answer

    print("\n====================")
    print("ANSWER")
    print("====================")

    print(answer)



if __name__ == "__main__":
    main()
