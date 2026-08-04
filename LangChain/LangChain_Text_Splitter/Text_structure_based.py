from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "This is a sample text that we will use to demonstrate the RecursiveCharacterTextSplitter. The splitter will break this text into smaller chunks based on the specified chunk size and overlap."

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

chunks = splitter.split_text(text)

print(chunks)
print("Number of chunks:", len(chunks))