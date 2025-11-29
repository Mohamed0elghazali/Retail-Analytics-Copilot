from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class MarkdownLoaderAndSplitter:
    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.docs = []
        self.chunks = []

        self._load_documents()
        self._chunk_documents()

    def _load_documents(self):
        docs = []
        for path in self.directory.glob("**/*.md"):
            loader = TextLoader(str(path), encoding="utf-8")
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source"] = str(path)
            docs.extend(file_docs)

        self.docs = docs

    def _chunk_documents(self):
        docs_chunks = []

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )

        for doc_idx, doc in enumerate(self.docs):
            chunk_id = 0
            split_docs = splitter.split_text(doc.page_content)

            for chunk in split_docs:
                docs_chunks.append(
                    Document(
                        page_content=chunk.page_content,
                        metadata={
                            "source": doc.metadata.get("source"),
                            "parent_id": doc_idx,
                            "chunk_id": chunk_id,
                            "headers": chunk.metadata,  
                        },
                    )
                )
                chunk_id += 1

        self.chunks = docs_chunks

class TfidfRetriever:
    def __init__(self, documents, k=5):
        """
        documents: list of langchain Documents
        """
        self.docs = documents
        self.k = k
        self.texts = [doc.page_content for doc in documents]

        # Build TF-IDF matrix
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def query(self, query_str: str, k: int = None):
        query_vec = self.vectorizer.transform([query_str])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        if not k:
            k = self.k

        ranked_idx = scores.argsort()[::-1][:k]

        results = []
        for idx in ranked_idx:
            doc = self.docs[idx]
            doc.metadata["score"] = round(float(scores[idx]), 4)
            results.append(doc)

        return results


if __name__ == "__main__":
    docs = MarkdownLoaderAndSplitter("docs")

    retriever = TfidfRetriever(docs.chunks, k=3)
    results = retriever.query("what is average order value ?", 10)

    for r in results:
        print(r.metadata)
        print(r.page_content)
        print("-" * 40)