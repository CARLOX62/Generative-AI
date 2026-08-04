import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# text = "This is a sample text that we will use to demonstrate the RecursiveCharacterTextSplitter. The splitter will break this text into smaller chunks based on the specified chunk size and overlap."

loader = PyPDFLoader("C:/Users/HP/Desktop/CODES/Generative Ai/LangChain/LangChain_Text_Splitter/Medical_book.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0,separators=["\n\n", "\n", " ", ""])

# result = splitter.split_text(text)

result = splitter.split_documents(docs)

print(result[1].page_content)
print("Number of chunks:", len(result))