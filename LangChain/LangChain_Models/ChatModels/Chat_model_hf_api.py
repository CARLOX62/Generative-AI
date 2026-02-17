from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text-generation",
)

result = llm.invoke("What is the capital of India?")
print(result)
