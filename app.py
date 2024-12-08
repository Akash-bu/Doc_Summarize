import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA 

def generate_response(upload_file, openai_api_key, query_text):

    if upload_file is not None:

        doc = [upload_file.read().decode()]

        text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0 )
        texts = text_splitter.create_documents(doc)

        embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)

        db = Chroma.from_documents(texts, embeddings)

        retriever = db.as_retriever()

        qa = RetrievalQA.from_chain_type(llm = OpenAI(openai_api_key = openai_api_key), chain_type = 'stuff', retriever = retriever)
        return qa.run(query_text)

res = []

upload_file = st.file_uploader("Upload a doc", type='txt')

query_text = st.text_input("Enter your question:", placeholder = 'Provide summary', disabled=not upload_file )


with st.form('form1', clear_on_submit = True):
    openai_api_key = st.text_input('Enter API key', type = 'password', disabled = not(upload_file and query_text))
    submitted = st.form_submit_button('Submit', disabled = not(upload_file and query_text) )

    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('loading...'):
            resp = generate_response(upload_file, openai_api_key, query_text)
            res.append(resp)
            del openai_api_key

if len(res):
    st.info(res)

