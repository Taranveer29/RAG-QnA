from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed to use sessions

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define directory for uploaded PDFs
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"

# Load the LangChain model
model = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL)
embeddings = OpenAIEmbeddings()
parser = StrOutputParser()

# Helper function to load PDF and create retriever
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()

    # Combine the text from all documents into a single string for summarization
    full_text = ' '.join(doc.page_content for doc in documents)

    # Generate a summary of the PDF
    summary_prompt = PromptTemplate.from_template(
        """
        Provide a summary of the following text in no more than 100 words:

        {text}
        """
    )
    summary_response = model.invoke(summary_prompt.format(text=full_text))
    summary = summary_response.content if hasattr(summary_response, 'content') else 'No summary found'

    # Create retriever for question answering
    vectorstore = DocArrayInMemorySearch.from_documents(documents, embedding=embeddings)
    return vectorstore.as_retriever(), summary

# Function to regenerate retriever from cached PDF
def regenerate_retriever(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    vectorstore = DocArrayInMemorySearch.from_documents(documents, embedding=embeddings)
    return vectorstore.as_retriever()

# Route for uploading PDFs and asking questions
@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    summary = None
    error = None

    if request.method == "POST":
        # Check if it's the PDF upload or the question submission
        if "pdf" in request.files:
            file = request.files["pdf"]
            if file.filename == "":
                error = "No selected file!"
            else:
                # Save the file and store its path in the session
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                session['pdf_path'] = file_path  # Store file path in session

                # Process the PDF and get summary
                retriever, summary = process_pdf(file_path)
                session['summary'] = summary  # Store summary in session

        elif "question" in request.form:
            question = request.form.get("question")
            if question and 'pdf_path' in session:
                # Regenerate the retriever based on the PDF path
                pdf_path = session['pdf_path']
                retriever = regenerate_retriever(pdf_path)
                summary = session.get('summary', None)

                # Generate response for the question
                prompt_template = PromptTemplate.from_template(
                    """
                    Answer the question based on the context below. If you can't answer the 
                    question, reply "I am sorry, it's out of my domain."

                    Context: {context}

                    Question: {question}
                    """
                )
                context = retriever.invoke(question)
                prompt = prompt_template.format(context=context, question=question)
                raw_response = model.invoke(prompt)
                answer = raw_response.content if hasattr(raw_response, 'content') else 'No answer found'
            else:
                error = "No question submitted or no PDF uploaded."
        else:
            error = "No file uploaded!"

    summary = session.get('summary', None)
    return render_template("index.html", answer=answer, summary=summary, error=error)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
