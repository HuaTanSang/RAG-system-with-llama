from rag import RAG 

def main(): 
    rag = RAG() 
    filepath = 'data.pdf'
    query = "summary this document"
    response = rag.run(filePath=filepath, query=query)

    print(response)

if __name__ == "__main__": 
    main() 
