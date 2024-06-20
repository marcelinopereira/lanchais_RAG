
import streamlit as st
import bs4
import chromadb
import tiktoken
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

# URL processing
def process_input(urls, question):
    model_local = Ollama(model="mistral")
    retriever = ""

    if (urls):
        # Convert string of URLs to list
        #file_path = ("{csv}")
        #urls_list = urls.split("\n")
        #docs = [WebBaseLoader(url).load() for url in urls_list]
        #docs_list = [item for sublist in docs for item in sublist]

        #split the text into chunks
        loader = WebBaseLoader(
        web_paths=([urls]),
        bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
def process_input(urls, question):
    model_local = Ollama(model="mistral")
    USER_AGENT = requests.utils.default_headers() 
    #convert urls to list
    loader = WebBaseLoader(
        web_paths=([urls]),
        bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
    ulr_list = urls.split("\n")
    docs = [loader.load() for url in ulr_list]
    docs_list = [item for sublist in docs for item in sublist]
    
    #split documents into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_split = text_splitter.split_documents(docs_list)
    
    #Convert text chunks into embeddings and store in vectorstore
    vectorstore = chromadb.from_documents(documents=doc_split, collection_name="rag_chroma", embedding = embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
                                        )
    
    retriever = vectorstore.as_retriever()    
    
    # perform the query RAG
    after_rag_tamplate = """responda a pergunta baseada somente no seguinte texto"""
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_tamplate)
    after_rag_chain = ({"Contexto": retriever, "pergunta": RunnablePassthrough()} 
                       |after_rag_prompt 
                       |model_local
                       |StrOutputParser())
    return after_rag_chain.invoke(question)

#streamlit app
st.title("LLM com Ollama")
st.write("Inclua as URLS, uma por linha")

# URLS 
Urls = st.text_area("URLS", height=200)
pergunta = st.text_input("Pergunta")

if st.button("Processar"):
    with st.spinner("Processando..."):
        resposta = process_input(Urls, pergunta)
        st.text_area("Resposta", value=resposta, height=300, disabled=True)