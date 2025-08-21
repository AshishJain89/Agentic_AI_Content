from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import models, QdrantClient
from tqdm import tqdm

def batch_iterate(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

class EmbedData:
    def __init__(self, 
                 embed_model_name="nomic-ai/nomic-embed-text-v1.5",
                 batch_size=32):
        
        self.embed_model_name = embed_model_name
        self.embed_model = self._load_embed_model()
        self.batch_size = batch_size
        self.embeddings = []

    def _load_embed_model(self):
        embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name,
                                           trust_remote_code=True,
                                           cache_folder='./hf_cache')
        return embed_model
    
    def generate_embedding(self, context):
        return self.embed_model.get_text_embedding_batch(context)
    
    def embed(self, contexts):
        self.contexts = contexts
        
        for batch_context in tqdm(batch_iterate(contexts, self.batch_size),
                                  total=len(contexts)//self.batch_size,
                                  desc="Embedding data in batches"):
                                  
            batch_embeddings = self.generate_embedding(batch_context)
            
            self.embeddings.extend(batch_embeddings)

class QdrantVDB:
    def __init__(self, collection_name, vector_dim=768, batch_size=512):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim

    def define_client(self):
        # Use in-memory client instead of requiring a server
        self.client = QdrantClient(":memory:")
        
    def create_collection(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(collection_name=self.collection_name,
                                          vectors_config=models.VectorParams(
                                              size=self.vector_dim,
                                              distance=models.Distance.DOT,
                                              on_disk=False),
                                          optimizers_config=models.OptimizersConfigDiff(
                                              default_segment_number=5,
                                              indexing_threshold=0)
                                         )
            
    def ingest_data(self, embeddata):
        for batch_context, batch_embeddings in tqdm(zip(batch_iterate(embeddata.contexts, self.batch_size), 
                                                        batch_iterate(embeddata.embeddings, self.batch_size)), 
                                                    total=len(embeddata.contexts)//self.batch_size, 
                                                    desc="Ingesting in batches"):
        
            # Create points for upsert
            points = []
            for i, (context, embedding) in enumerate(zip(batch_context, batch_embeddings)):
                points.append(models.PointStruct(
                    id=i,
                    vector=embedding,
                    payload={"context": context}
                ))
            
            # Use upsert instead of upload_collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        self.client.update_collection(collection_name=self.collection_name,
                                    optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
                                    )

class Retriever:
    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query):
        query_embedding = self.embeddata.embed_model.get_text_embedding(query)

        # select the top 3 results
        result = self.vector_db.client.search(
            collection_name=self.vector_db.collection_name,
            query_vector=query_embedding,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=True,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
            limit=3,
            timeout=1000,
        )

        context = [dict(data) for data in result]
        combined_prompt = []

        for entry in context[:3]:
            context = entry["payload"]["context"]
            combined_prompt.append(context)

        final_output = "\n\n---\n\n".join(combined_prompt)
        return final_output

# Example usage
if __name__ == "__main__":
    # Sample FAQ data
    faq_text = """Question 1: What is the first step before building a machine learning model?
Answer 1: Understand the problem, define the objective, and identify the right metrics for evaluation.

Question 2: How important is data cleaning in ML?
Answer 2: Extremely important. Clean data improves model performance and reduces the chance of misleading results.

Question 3: Should I normalize or standardize my data?
Answer 3: Yes, especially for models sensitive to feature scales like SVMs, KNN, and neural networks."""

    new_faq_text = [i.replace("\n", " ") for i in faq_text.split("\n\n")]
    
    # Create embeddings
    batch_size = 32
    embeddata = EmbedData(batch_size=batch_size)
    embeddata.embed(new_faq_text)
    
    # Setup database
    database = QdrantVDB("ml_faq_collection")
    database.define_client()
    database.create_collection()
    database.ingest_data(embeddata)
    
    # Search
    result = Retriever(database, embeddata).search("How to prevent overfitting?")
    print(result)
