import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import time


with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")
    
# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
llm = ChatGroq(model='Deepseek-R1-Distill-Llama-70b', groq_api_key=groq_api_key)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

st.title("📚 Smart Study Recommendation System")

st.header("Step 1: Add Your Personal Study Information")

user_study_history = st.text_area("Your study history (topics studied):")
user_assessment_performance = st.text_area("Your assessment performance (e.g., scores, weak topics):")
user_exam_schedule = st.text_area("Your upcoming exam schedule (dates and subjects):")
user_weak_points = st.text_area("Topics you feel weak in:")

if st.button("Prepare My Study Profile"):
    # Create a single document from all user info
    user_text = f"""
Study History: {user_study_history}
Assessment Performance: {user_assessment_performance}
Exam Schedule: {user_exam_schedule}
Weak Points: {user_weak_points}
"""
    user_doc = Document(page_content=user_text)

    # Split user doc into chunks (if needed)
    split_docs = text_splitter.split_documents([user_doc])

    # Build vector store from user docs only
    vectors = FAISS.from_documents(split_docs, embeddings)
    retriever = vectors.as_retriever()

    st.session_state['vectors'] = vectors
    st.session_state['retriever'] = retriever

    st.success(" Your study profile has been processed! Now you can ask your study questions below.")

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    '''You are an intelligent academic assistant helping students study more effectively. 
Based on the student's study history, performance in assessments, exam schedule, and identified weak points, 
recommend what they should study next.

Only suggest topics that are relevant and aligned with their upcoming exams and areas where they need improvement.

Use the following context extracted from student records:
{context}

Give a clear and prioritized list of study recommendations. Start with the most urgent based on the exam schedule and weakest performance areas. 
Mention the topic name only, and avoid explanations unless necessary.

Student query: "{input}"'''
)

st.header("Step 2: Ask Your Study Question")

user_prompt = st.text_input("Ask your study question:")

if user_prompt:
    if 'retriever' not in st.session_state:
        st.warning("Please enter and prepare your study information first.")
    else:
        with st.spinner("Analyzing your study profile and preparing recommendation..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever_chain = create_retrieval_chain(st.session_state.retriever, document_chain)

            start = time.process_time()
            response = retriever_chain.invoke({'input': user_prompt})
            duration = time.process_time() - start

            st.success(f"Response generated in {duration:.2f} seconds")
            st.markdown("### Recommendation:")
            st.write(response['answer'])

            with st.expander('📄 Document similarity search (Context Used)'):
                for i, doc in enumerate(response.get('context', [])):
                    st.write(doc.page_content)
                    st.write('---')
