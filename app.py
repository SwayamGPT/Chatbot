import os
import streamlit as st
import tempfile
import uuid
from dotenv import load_dotenv
from PIL import Image
import pytesseract

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

# --- Configuration ---
load_dotenv()
FALLBACK_MODEL = "mistralai/mistral-7b-instruct:free"

# For Windows users, you might need to set the Tesseract path explicitly.
# Uncomment and update the line below if you get a "Tesseract not found" error.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# --- Model and API Initialization ---
try:
    local_llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 200},
    )
except Exception as e:
    st.error(f"Could not load HuggingFace model: {e}")
    local_llm = None

try:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
except Exception as e:
    st.error(f"Failed to initialize OpenRouter client: {e}")
    openrouter_client = None

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Could not initialize embeddings: {e}")
    embeddings = None


# --- Base Knowledge ---
document_content = """
LangChain is a framework for developing applications powered by language models.
Retrieval-Augmented Generation (RAG) enhances LLMs by fetching facts from external knowledge bases.
A vector database like FAISS stores and queries numerical representations (vectors) of data.
"""
file_path = "knowledge_base.txt"
with open(file_path, "w", encoding="utf-8") as f:
    f.write(document_content)


# --- Prompts ---
contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question. Do NOT answer the question, just reformulate it if needed."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)

qa_system_prompt = """You are a helpful and friendly AI assistant. Your main job is to answer questions based on the context provided.
If the context doesn’t contain the answer, you MUST reply with the exact phrase: "I do not have that information in the context."

Context:
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)


# --- Helper Functions (Updated for Streaming) ---
def ask_openrouter_stream(prompt: str, model=FALLBACK_MODEL):
    """Streaming fallback function."""
    if not openrouter_client:
        yield "OpenRouter client is not initialized."
        return

    try:
        stream = openrouter_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""
    except Exception as e:
        yield f"Error with OpenRouter: {e}"

def get_text_from_images(image_files):
    full_text = ""
    for image_file in image_files:
        try:
            img = Image.open(image_file)
            text = pytesseract.image_to_string(img)
            full_text += f"\n\n--- Content from {image_file.name} ---\n{text}"
        except Exception as e:
            st.error(f"Error processing image {image_file.name}: {e}")
    return full_text

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# --- Core RAG Chain Creation Logic ---
def create_rag_chain(documents):
    if not local_llm or not embeddings or not documents:
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    try:
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    history_aware_retriever = create_history_aware_retriever(local_llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(local_llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

def get_answer_stream(user_input: str, session_id: str):
    """
    Gets a full answer from the RAG chain and streams the fallback if needed.
    This is a hybrid approach to fix the bug where both messages appeared.
    """
    if "conversational_rag_chain" not in st.session_state or st.session_state.conversational_rag_chain is None:
        yield "The chatbot is not initialized. Please check the model and embedding configurations."
        return

    chain = st.session_state.conversational_rag_chain
    
    # Get the full response from the local RAG chain (it's fast)
    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    answer = response.get("answer", "")

    # Check the response
    if "I do not have that information in the context" in answer:
        # If not found, show an info message and then stream from the fallback
        st.info("The answer was not found in the provided knowledge. Asking a general model...")
        yield from ask_openrouter_stream(user_input)
    else:
        # If found, "stream" the whole answer at once.
        # st.write_stream can handle a generator that yields a single string.
        yield answer

    chain = st.session_state.conversational_rag_chain
    
    full_response = ""
    is_not_found = False

    for chunk in chain.stream({"input": user_input}, config={"configurable": {"session_id": session_id}}):
        if "answer" in chunk:
            answer_chunk = chunk["answer"]
            full_response += answer_chunk
            if "I do not have that information in the context" in full_response:
                is_not_found = True
                break
            yield answer_chunk



# --- Streamlit UI Setup ---
st.set_page_config(page_title="Chatbot", layout="centered")
st.title("Ask me anything!")

# Initialize Session State and Default Chain
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "conversational_rag_chain" not in st.session_state:
    try:
        default_loader = TextLoader(file_path)
        default_docs = default_loader.load()
        st.session_state.conversational_rag_chain = create_rag_chain(default_docs)
    except Exception as e:
        st.error(f"Failed to create default chatbot: {e}")
        st.session_state.conversational_rag_chain = None


# --- Sidebar for File Upload and Processing ---
with st.sidebar:
    st.title("Upload & Process")
    st.info("For image processing, ensure you have Google's Tesseract OCR engine installed on your system.")
    
    uploaded_files = st.file_uploader(
        "Upload documents to add them to the knowledge base",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            all_docs = []
            try:
                text_loader = TextLoader(file_path)
                all_docs.extend(text_loader.load())
            except Exception as e:
                st.warning(f"Could not reload base knowledge file: {e}")

            pdf_files = [f for f in uploaded_files if f.type == "application/pdf"]
            image_files = [f for f in uploaded_files if f.type in ["image/png", "image/jpeg"]]

            for pdf_file in pdf_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_file.read())
                    pdf_loader = PyPDFLoader(tmp.name)
                    all_docs.extend(pdf_loader.load())
                os.remove(tmp.name)
            
            if image_files:
                image_text = get_text_from_images(image_files)
                if image_text:
                    all_docs.append(Document(page_content=image_text))
            
            st.session_state.conversational_rag_chain = create_rag_chain(all_docs)
            if st.session_state.conversational_rag_chain:
                st.success("Documents processed successfully!")
            else:
                st.error("Failed to process documents.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if "session_id" in st.session_state and st.session_state.session_id in store:
            del store[st.session_state.session_id]
        st.rerun()


# --- Main Chat Interface (Updated for Streaming) ---
if not st.session_state.conversational_rag_chain:
    st.warning("Chatbot is not available. Please check the logs for errors.")
else:
    st.write("Ask me anything!")

    for role, content in st.session_state.messages:
        avatar = "❓" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append(("user", prompt))
        with st.chat_message("user", avatar="❓"):
            
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            greetings = ["hi", "hello", "hey", "hola"]
            if prompt.lower().strip() in greetings:
                response = "Hello! 👋 How can I help you today?"
                st.write(response)
            else:
                response_generator = get_answer_stream(prompt, st.session_state["session_id"])
                response = st.write_stream(response_generator)
            
            st.session_state.messages.append(("assistant", response))
