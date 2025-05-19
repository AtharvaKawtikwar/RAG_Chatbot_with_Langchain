import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import sys

def check_environment():
    """Check for required packages and installations."""
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("\nError: PyTorch not installed. Please install it first.")
        sys.exit(1)

def load_and_process_data():
    """Load and preprocess the books dataset."""
    try:
        df = pd.read_csv('books.csv')
        print("\nDataset preview:")
        print(df.head())
        
        df['content'] = df.apply(
            lambda row: (
                f"Title: {row['Title']}\n"
                f"Author: {row['Author']}\n"
                f"Genre: {row['Genre']}\n"
                f"Publisher: {row['Publisher']}\n"
                f"Height: {row['Height']}"
            ),
            axis=1
        )
        
        loader = DataFrameLoader(df, page_content_column="content")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        sys.exit(1)

def setup_embeddings():
    """Set up local embeddings."""
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        print(f"\nError loading embeddings: {str(e)}")
        sys.exit(1)

def setup_llm():
    """Set up local LLM with fallback options."""
    try:
        # Try to use a small local model
        model_name = "gpt2"  # Small model that works without GPU
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            temperature=0.7
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        print(f"\nError setting up LLM: {str(e)}")
        sys.exit(1)

def setup_rag_pipeline():
    """Set up the complete RAG pipeline."""
    print("\nSetting up RAG pipeline...")
    docs = load_and_process_data()
    embeddings = setup_embeddings()
    
    print("\nCreating vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("books_index")
    
    print("\nSetting up LLM...")
    llm = setup_llm()
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

def chat_with_bot(qa_chain):
    """Interactive chat interface."""
    print("\nWelcome to the Local Book Recommendation Chatbot!")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['quit', 'exit']:
                break
            
            result = qa_chain.invoke({"query": query})  # Updated to use invoke()
            print("\nBot:", result["result"])
            
            if result["source_documents"]:
                print("\nRecommended Books:")
                for i, doc in enumerate(result["source_documents"], 1):
                    metadata = doc.metadata
                    print(f"{i}. {metadata.get('Title', 'Unknown')}")
                    print(f"   Author: {metadata.get('Author', 'Unknown')}")
                    print(f"   Genre: {metadata.get('Genre', 'Unknown')}")
                    print(f"   Publisher: {metadata.get('Publisher', 'Unknown')}\n")
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    check_environment()
    try:
        qa_chain = setup_rag_pipeline()
        print("\nSystem ready! Ask about books...")
        chat_with_bot(qa_chain)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")