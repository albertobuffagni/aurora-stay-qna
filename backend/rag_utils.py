
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
import os
import re

# Split text into lowercase words for BM25 keyword matching
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$]+", text.lower())


# Glossary: guest wording -> policy wording, added to the query before searching
SYNONYMS = {
    r"\b(loud|noise|noisy|party|music)\b": "noise disturbance quiet hours complaint",
    r"\b(leave|stay|check ?out)\b.*\b(late|later|\d{1,2} ?pm)\b|\blate check ?out\b": "late departure late checkout fee",
    r"\b(arrive|check ?in)\b.*\b(early|\d{1,2} ?am)\b|\bearly check ?in\b": "early arrival",
    r"\b(ota|expedia|booking\.com|travel agency|third[- ]party)\b": "online travel agency",
    r"\b(hold|pending|authorization|deposit)\b": "incidentals authorization release",
    r"\b(smok|vap)": "smoking vaping remediation charge",
}

# Property rows look like "Property: ASH-DEN Union, Denver CO; ..." -> map "denver" to "ASH-DEN Union"
PROPERTY_ROW = re.compile(r"Property: (ASH-[A-Z]+ [A-Za-z ]+?), ([A-Za-z ]+) [A-Z]{2};")

def build_property_map(chunks) -> dict:
    property_map = {}
    for c in chunks:
        m = PROPERTY_ROW.search(c.page_content)
        if m:
            property_map[m.group(2).strip().lower()] = m.group(1)
    return property_map


def expand_query(query: str, property_map: dict) -> str:
    q, extra = query.lower(), []

    # City or hotel name -> property code, only when the code is not already in the question
    for city, prop in property_map.items():
        code, name = prop.split()[0].lower(), prop.split()[1].lower()
        if (city in q or name in q) and code not in q:
            extra.append(prop)

    for pattern, terms in SYNONYMS.items():
        if re.search(pattern, q):
            extra.append(terms)

    return query + (" (" + "; ".join(extra) + ")" if extra else "")


# Build the BM25 keyword index and the property map once per vector store, over the same chunks stored in FAISS
_bm25_cache = {}

def get_bm25(vectorstore):
    if id(vectorstore) not in _bm25_cache:
        chunks = [
            vectorstore.docstore.search(vectorstore.index_to_docstore_id[i])
            for i in range(vectorstore.index.ntotal)
        ]
        bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])
        _bm25_cache[id(vectorstore)] = (bm25, chunks, build_property_map(chunks))
    return _bm25_cache[id(vectorstore)]


# Retrieve the top-k relevant chunks with hybrid search (semantic + keyword, merged with RRF)
def retrieve(query: str, k: int = 2, vectorstore=None, n_candidates: int = 20, rrf_c: int = 60) -> list[str]:
    bm25, chunks, property_map = get_bm25(vectorstore)

    # Add policy wording and property codes to the question (the original question is kept)
    query = expand_query(query, property_map)

    # Semantic ranking from FAISS
    semantic = vectorstore.similarity_search(query, k=n_candidates)

    # Keyword ranking from BM25
    scores = bm25.get_scores(tokenize(query))
    keyword = [chunks[i] for i in sorted(range(len(chunks)), key=lambda i: -scores[i])[:n_candidates]]

    # Reciprocal Rank Fusion: add 1 / (rrf_c + rank) from each ranking
    fused, docs_by_key = {}, {}
    for ranking in (semantic, keyword):
        for rank, doc in enumerate(ranking):
            key = (doc.metadata.get("source"), doc.page_content)
            fused[key] = fused.get(key, 0) + 1 / (rrf_c + rank + 1)
            docs_by_key[key] = doc

    results = [docs_by_key[key] for key in sorted(fused, key=fused.get, reverse=True)[:k]]

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    rules = """
    1. Use ONLY the information in the Retrieved Policy Information. Do not use outside knowledge or assumptions.
    2. Answer the specific question directly in the first sentence, then add only the supporting details needed.
    3. Include every relevant amount, time, deadline, property code and limit exactly as written in the sources.
    4. Include the conditions, exceptions and approvals that apply (e.g. "subject to availability",
       "the reservation confirmation controls", service animals vs. pets).
    5. If the question mentions a specific property (e.g. ASH-MIA Brickell), only give values for that property.
       If it mentions several properties, answer for each one separately.
    6. Do not include policy information that is not related to the question.
    7. Do not invent fees, refund rules, times or procedures that are not in the retrieved information.
    8. If the retrieved information is insufficient to answer fully, say exactly what is missing:
       "The available policy documents do not contain enough information to answer this question."
       If only part of the answer is supported, answer that part and state what could not be confirmed.
    9. If the retrieved sources conflict, mention the conflict instead of choosing one.
    10. End with a short "Source:" line naming the document(s) the answer came from.
    11. Keep the answer concise, professional and easy for hotel staff to read to a guest.
    """

    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    {rules}

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 5,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
