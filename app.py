import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Page configuration
st.set_page_config(page_title="Book Recommendation Chatbot")

@st.cache_resource
def load_qa_chain():
    """Load the QA chain with caching for Streamlit."""
    try:
        # Using local embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load vector store
        vectorstore = FAISS.load_local("books_index", embeddings)
        
        # Setup local LLM (small GPT-2 model)
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            temperature=0.7
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat interface
st.title("📚 Book Recommendation Chatbot")
st.write("Ask me about books and I'll recommend some great reads!")

if prompt := st.chat_input("What kind of books are you looking for?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        # Get bot response
        qa_chain = load_qa_chain()
        result = qa_chain.invoke({"query": prompt})
        response = result["result"]
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Show sources if available
        if result["source_documents"]:
            with st.expander("Recommended Books"):
                for doc in result["source_documents"]:
                    meta = doc.metadata
                    st.write(f"**{meta.get('Title', 'Unknown')}**")
                    st.write(f"Author: {meta.get('Author', 'Unknown')}")
                    st.write(f"Genre: {meta.get('Genre', 'Unknown')}")
                    st.write("---")
                    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")