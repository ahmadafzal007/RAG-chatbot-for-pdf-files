import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv



# Set the environment variable
os.environ['GOOGLE_API_KEY'] = '*************'

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
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

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()






# import tkinter as tk
# from tkinter import filedialog, messagebox, scrolledtext
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# import PyPDF2
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# # Load environment variables if any
# load_dotenv()

# # Initialize Google Generative AI
# from google.generativeai import configure
# import google.generativeai as genai
# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# # Initialize tkinter window
# root = tk.Tk()
# root.title("MedTalk")

# # Function to sanitize text
# def sanitize_text(text):
#     # Replace or remove surrogate pairs
#     return text.encode('utf-16', 'surrogatepass').decode('utf-16')

# # Function to get text from PDFs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         try:
#             pdf_reader = PyPDF2.PdfReader(pdf)
#             for page in pdf_reader.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     sanitized_text = sanitize_text(page_text)
#                     text += sanitized_text
#         except Exception as e:
#             print(f"Error processing PDF: {e}")
#             continue
#     return text

# # Function to split text into chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# # Function to create vector store
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # Function to initialize conversational chain
# def get_conversational_chain():
#     prompt_template = '''
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not
#     in the provided context redirect to the gemini llm and and respond me, you must give me answer at every cost\n\n
#     Context:\n {context} \n
#     Question: \n {question}\n

#     Answer:

#     '''
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# # Function to handle user input and generate response
# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     response_text.insert(tk.END, f"Reply: {response['output_text']}\n\n")

# # Function to handle file upload and processing
# def process_files():
#     file_paths = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
#     if file_paths:
#         messagebox.showinfo("Files Uploaded", "PDF files uploaded successfully!")
#         raw_text = get_pdf_text(file_paths)
#         text_chunks = get_text_chunks(raw_text)
#         get_vector_store(text_chunks)
#         response_text.insert(tk.END, "Processing complete.\n")

# # GUI setup
# label = tk.Label(root, text="Ask a question from your PDF files:")
# label.pack()

# question_entry = tk.Entry(root, width=50)
# question_entry.pack()

# submit_button = tk.Button(root, text="Submit & Process", command=lambda: user_input(question_entry.get()))
# submit_button.pack()

# file_label = tk.Label(root, text="Upload your PDF files:")
# file_label.pack()

# file_button = tk.Button(root, text="Upload", command=process_files)
# file_button.pack()

# response_text = scrolledtext.ScrolledText(root, width=80, height=20)
# response_text.pack()

# root.mainloop()
