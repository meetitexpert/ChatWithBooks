import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def process_files(uploaded_files):
    documents = []
    documents.extend(process_pdf(uploaded_files))
    
    return documents

def process_pdf(file):
    loader = PyPDFLoader(file)
    docs = loader.load()
    return docs


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(uploaded_files):
    documents = process_pdf(uploaded_files)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
        
    # Generate embeddings and create FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    # chunks = get_text_chunks(texts)
    
    # docs covert into vector database
    vector_store = FAISS.from_documents(docs, embeddings)
    
    #vector database save locally
    vector_store.save_local("faiss_index",)


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details inside the context, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer,\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()

    best_result = docs[0]
    print('--------------------------RESULT----------------')
    print(docs[0].page_content)
    response = chain.invoke(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    print(response)
    result = response["output_text"]
        
    st.write("\nPage NO:" + str(best_result.metadata["page"]) )
    st.write("\nSource:" + best_result.metadata["source"] )
    st.write(result )


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDFs💁")
    

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", type="pdf")
        if pdf_docs:
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, pdf_docs.name)
            with open(path, "wb") as f:
                f.write(pdf_docs.getvalue())
                if st.button("Submit & Process"):
                    with st.spinner("Processing..."):
                        get_vector_store(path)
                        st.success("Done")
        
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
       
        st.write('Created by: B-Technos')
            



if __name__ == "__main__":
    main()
