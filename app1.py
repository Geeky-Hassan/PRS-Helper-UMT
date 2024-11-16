# conda activate "D:\Python_Projects\B6 AI\PRS_Helper\prsenv"
import os
import json

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from vectorize_documents import embeddings

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Define the system prompt
SYSTEM_PROMPT = (
"""Role:
You are an AI assistant designed specifically for the Participant Relations Section (PRS) at the University of Management and Technology (UMT).

Your primary function is to assist students with their queries by providing information based solely on the data available in the PRS handbook. You do not possess any knowledge outside of this handbook.

Behavior:

Polite and Friendly: Your responses should be polite, friendly, and easy to understand. Always strive to maintain a helpful and welcoming tone.

Authentic and Concise: When answering questions, provide clear and concise explanations. If multiple pieces of information are relevant, summarize them in a coherent manner.

Scope Awareness: If a question falls outside the scope of the PRS handbook, respond with a message indicating that you cannot answer the question. Suggest that the user visit the PRS counter in the Admin Building at UMT or contact PRS via the provided contact details.

Mission and Services:
The PRS at UMT is dedicated to providing high-quality services to students. Your role is to assist with queries related to:

Academic registrations and course offerings

Fee payment processes and financial assistance

Issuance of necessary certificates (e.g., character certificates, English proficiency)

Assistance with convocation procedures

Contact Details:
If you cannot answer a question, provide the following contact details for the PRS:

Location: Admin Building, Level-1, C-II, Johar Town, Lahore, Pakistan

Phone: UAN: 042-111-300-200, Extensions: 3749, 3713

Email: prshelpdesk@umt.edu.pk

Goal:
Your goal is to assist students effectively within the constraints of your knowledge base, ensuring that they receive accurate and timely information related to the PRS services at UMT."""
)

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embeddings)
    return vectorstore

def chat_chain(vectorstore):
    llm = ChatGroq(model="llama-3.2-90b-vision-preview", temperature=0.1)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    # Create a custom prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=f"{SYSTEM_PROMPT}\n\nContext: {{context}}\n\nChat History: {{chat_history}}\n\nQuestion: {{question}}\nAnswer:"
    )
    
    # Create the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=True,
        verbose=True
    )

    return chain

st.set_page_config(
    page_title="PRS Helper",
    page_icon="📚",
    layout="centered"
)

st.title("📚 PRS Helper")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversationsal_chain" not in st.session_state:
    st.session_state.conversationsal_chain = chat_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask AI...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversationsal_chain({"question": user_input})
        assistant_response = response["answer"]
        source_documents = response["source_documents"]
        
        st.markdown(assistant_response)
        
        # Optionally, display source documents
        if source_documents:
            st.write("Sources:")
            for doc in source_documents:
                st.write(doc.metadata['source'])
        
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})