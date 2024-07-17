import boto3
import streamlit as st
import os
import uuid
from PIL import Image
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import random

# Set page configuration
st.set_page_config(page_title="PDF Processor & QA", page_icon="📄", layout="centered")

# Add custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stFileUploader label {
            color: #555555;
        }
        .stSpinner {
            margin: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "vector_store_created" not in st.session_state:
    st.session_state.vector_store_created = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# s3_client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="ap-northeast-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def get_unique_id():
    return str(uuid.uuid4())

# SPLIT THE PAGES / TEXT INTO CHUNKS
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(request_id, documents):
    if documents is None or len(documents) == 0:
        st.error("Documents cannot be None or empty.")
        return False
    
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    file_name = f"{request_id}.bin"
    folder_path = "/tmp"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    # upload to s3
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True

def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"/tmp/my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"/tmp/my_faiss.pkl")

def get_llm():
    llm = Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample':512})
    return llm

def get_response(llm, vectorstore, question):
    prompt_template = """
    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": question})
    return answer['result']

def process_pdf(uploaded_file):
    request_id = get_unique_id()
    saved_file_name = f"{request_id}.pdf"
    with open(saved_file_name, mode="wb") as w:
        w.write(uploaded_file.getvalue())
    
    loader = PyPDFLoader(saved_file_name)
    pages = loader.load_and_split()
    if pages is None or len(pages) == 0:
        st.error("No pages loaded from the PDF.")
        return

    st.write(f"**Total pages:** {len(pages)}")

    # Split Text
    splitted_docs = split_text(pages, 1000, 200)
    if splitted_docs is None or len(splitted_docs) == 0:
        st.error("No documents were generated after splitting.")
        return

    st.write(f"**Splitted Docs length:** {len(splitted_docs)}")
    st.write("### Sample Documents")
    random_indices = random.sample(range(len(splitted_docs)), 2)
    st.write(splitted_docs[random_indices[0]])
    st.write(splitted_docs[random_indices[1]])

    with st.spinner("Creating the vector store..."):
        result = create_vector_store(request_id, splitted_docs)

    if result:
        st.session_state.vector_store_created = True
        st.success("Your PDF is processed successfully. You can now ask questions about it.")
    else:
        st.error("Error!! Please check logs.")

def main():
    st.title("PDF Processor & QA 📄")
    st.markdown("### Upload a PDF file to process or ask questions")

    # Adding an image banner
    image = Image.open("pdf_processing.jpeg")
    st.image(image, use_column_width=True)

    # File upload section
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    # Visible help text
    st.markdown(
        "<p style='color: #555555;'>Ensure that the pages in the PDF are not image-based text. They should allow for copying and pasting.</p>",
        unsafe_allow_html=True
    )
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        if not st.session_state.vector_store_created:
            process_pdf(uploaded_file)

    if st.session_state.vector_store_created:
        # Show the uploaded file name
        st.write(f"**Uploaded file:** {st.session_state.uploaded_file.name}")

        # Question asking section
        st.markdown("### Ask Questions about the PDF")

        question = st.text_input("Please ask your question", key="question", help="Type your question here...", placeholder="Enter your question...")

        if st.button("Ask question"):
            if len(question.strip()) == 0:  # Check if the question is empty
                st.warning("Please enter a question before submitting.")
            else:
                with st.spinner("Querying..."):
                    load_index()

                    faiss_index = FAISS.load_local(
                        index_name="my_faiss",
                        folder_path="/tmp",
                        embeddings=bedrock_embeddings,
                        allow_dangerous_deserialization=True
                    )

                    llm = get_llm()
                    response = get_response(llm, faiss_index, question)
                    
                    st.session_state.last_question = question
                    st.session_state.last_response = response

        if st.session_state.last_question:
            st.write("Your question: ", st.session_state.last_question)
            st.write(st.session_state.last_response)

if __name__ == "__main__":
    main()
