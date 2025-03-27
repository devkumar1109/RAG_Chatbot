from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

st.title('Q&A Conversational RAG Chatbot')
st.write('Upload your PDFs and ask any question related to it')
groq_api_key = st.text_input('Enter your GROQ API key', type = 'password')
st.sidebar.title('Settings')
temperature = st.sidebar.slider('Temperature', min_value=0., max_value=1., value = 0.4)
max_tokens = st.sidebar.slider('Max_Tokens', min_value=100, max_value=1000, value = 200)


## Prompt for taking context, chat history and query from user
prompt = ChatPromptTemplate.from_messages([
                             ("system", 'You are an AI chatbot. Please answer the following question in detail based on the context and chat history. Also use your own knowledge wherever required'),
                             ('human', "<Context>{context}</Context>"),
                             ('human', "<History>{chat_history}</History>"),
                             ('human', "{input}")
])


if groq_api_key:
    session_id = st.text_input('Session-ID', value = 'default_session')
    llm = ChatGroq(model='llama-3.3-70b-specdec', groq_api_key=groq_api_key)
    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

    uploaded_files = st.file_uploader('Please upload your pdf file', type = 'pdf', accept_multiple_files = True)
    
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, 'wb') as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 3000, chunk_overlap = 300)
        splits = text_splitter.split_documents(documents)
        vectordb = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectordb.as_retriever()
        
        # usecase(csdc) - for this func it is compulsory to have context named variable in the prompt
        # this converts the document to string and passes to the context variable 
        document_chain = create_stuff_documents_chain(llm,prompt)
        # usecase(crc) - based on similarity search of question in retriever it return the relevant documents
        # and passes it to document chain
        retrieval_chain = create_retrieval_chain(retriever,document_chain)
        
        if 'store' not in st.session_state:
            st.session_state.store = {}
        
        def get_session_history(session_id:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]   
        
        user_input = st.text_input('Please ask any question from the pdf')
        config = {'configurable':{'session_id':session_id}}
        rag_chain = RunnableWithMessageHistory(retrieval_chain,get_session_history,
                                               input_messages_key='input',
                                               output_messages_key='answer',
                                               history_messages_key='chat_history')  
        
        if user_input:
            result = rag_chain.invoke({'input':user_input}, config = config)
            st.write('The answer is: \n\n\n', result['answer'])
            history = st.sidebar.write(result['context'])

# ENTIRE WORKFLOW - 
# S1: Rag chain invoked - input taken from user and goes to retrieval chain
# S2: Retrieval chain retrieves the documents similiar to the input and sents to stuff_documents_chain
# S3: In SDC, documents are converted to string by reading their content and passed to the prompt as context
# S4: user input goes to the input variable in prompt 
# S5: from rag_chain using session id the history is passed to the chat history variable in prompt
# S6: SDC calls the llm with this entire prompt and return the results.
        
        
        
        
            