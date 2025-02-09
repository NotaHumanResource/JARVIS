import os
import sqlite3
import PyPDF2
import docx
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize

# Manually set the NLTK data path to where 'punkt' is located
nltk.data.path.append("C:/Users/kenba/AppData/Roaming/nltk_data")

# Ensure required resources are available
nltk.download('punkt', quiet=True)

# Database path
db_path = 'jarvis_data.db'

def init_database():
    """Initialize the database with the required tables if they do not exist."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            category TEXT,
            relevance_score REAL
        )
    ''')
    conn.commit()
    conn.close()

def read_pdf(file_path):
    """Read text from a PDF file using the updated PyPDF2 PdfReader."""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)  # ✅ Use PdfReader instead of PdfFileReader
        text = ""
        for page in reader.pages:  # ✅ reader.pages replaces reader.getPage()
            text += page.extract_text() or ""  # ✅ Use extract_text() safely
    return text

def read_docx(file_path):
    """Read text from a DOCX file."""
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def read_txt(file_path):
    """Read text from a TXT file."""
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def chunk_text(text, max_length=500):
    """Breaks long text into smaller chunks for better processing."""
    sentences = sent_tokenize(text)  # Tokenize the text into sentences once
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_length:  # +1 for the space
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks


def store_document_knowledge(file_path, text, model, conn):
    """Store extracted knowledge from a document into the database."""
    sentences = sent_tokenize(text)
    num_stored = 0
    for sentence in sentences:
        c = conn.cursor()
        c.execute("""
            INSERT INTO memories (memory, source, timestamp, category, relevance_score)
            VALUES (?, ?, ?, ?, ?)
        """, (sentence, file_path, datetime.now(), "Document", 0.8))
        conn.commit()
        num_stored += 1
    return num_stored

def retrieve_relevant_knowledge(topic, conn):
    """Retrieve relevant knowledge from the database based on the topic."""
    c = conn.cursor()
    c.execute("""
        SELECT memory, source, timestamp 
        FROM memories 
        WHERE memory LIKE ?
        ORDER BY relevance_score DESC, timestamp DESC
    """, ('%' + topic + '%',))
    return c.fetchall()

def main():
    """Main function to run the Streamlit application."""
    init_database()
    
    st.title("JARVIS")

    st.write("""
    ## Reminder:
    - Use the phrase "remember that" to store important information
    - Use the phrase "forget that" to delete the displayed entry from the database
    - Use the phrase "summarize" to get a summary of the 3 most recent memories
    - Use the phrase "read [file name]" to read a local document
    - Use the phrase "what do you know about [topic]" to retrieve relevant knowledge
    """)

    system_prompt = """
    You have expertise in technology, science, and problem-solving. 
    You have a good sense of humour.
    Your purpose is to be truth-seeking, helpful, precise, and analytical.
    You seek to remember important details from conversations to learn and become more knowledgable.

    You excel at:
    - Providing detailed technical analysis and solutions
    - Managing complex systems and information
    - Offering clear, direct communication
    - Learning from interactions to provide more relevant assistance
    - Keeping track of important information from our conversations.
    """

    template = f"""{system_prompt}

    Context: {{context}}
    Question: {{question}}

    Answer: Let's think step by step."""

    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model="llama3:latest")
    chain = prompt | model  # Create the chain once
    
    question = st.chat_input("Enter your question here")
    
    if question:
        conn = sqlite3.connect(db_path)
        
        try:
            if question.lower().startswith("read "):
                file_name = question[5:].strip()
                file_path = os.path.join('C:\\Users\\kenba\\source\\repos\\Ollama3\\LocalDocs', file_name)
                if os.path.exists(file_path):
                    if file_name.endswith('.pdf'):
                        text = read_pdf(file_path)
                    elif file_name.endswith('.docx'):
                        text = read_docx(file_path)
                    elif file_name.endswith('.txt'):
                        text = read_txt(file_path)
                    else:
                        st.error("Unsupported file format.")
                        text = ""
                    
                    if text:
                        st.write(f"Processing {file_name}...")
                        num_stored = store_document_knowledge(file_path, text, model, conn)
                        st.write(f"Extracted and stored {num_stored} pieces of information from the document.")
                        st.write("Content preview:")
                        st.write(text[:500] + "...")
                else:
                    st.error("File not found.")
            
            elif question.lower().startswith("what do you know about "):
                topic = question[20:].strip()
                relevant_knowledge = retrieve_relevant_knowledge(topic, conn)
                context = "\n".join([f"From {k[1]} ({k[2]}): {k[0]}" for k in relevant_knowledge])
                result = chain.invoke({"context": context, "question": question})
                st.write(f"Answer: {result}")
            
            else:
                relevant_knowledge = retrieve_relevant_knowledge(question, conn)
                context = "\n".join([f"From {k[1]} ({k[2]}): {k[0]}" for k in relevant_knowledge])
                result = chain.invoke({"context": context, "question": question})
                st.write(f"Answer: {result}")

                if "remember that" in question.lower():
                    c = conn.cursor()
                    c.execute("""
                        INSERT INTO memories (memory, source, timestamp, category, relevance_score)
                        VALUES (?, ?, ?, ?, ?)
                    """, (result, "user_input", datetime.now(), "General Knowledge", 0.9))
                    conn.commit()
                    st.write(f"JARVIS remembers: {result}")
                
                elif "summarize" in question.lower():
                    c = conn.cursor()
                    c.execute("""
                        SELECT memory, source, timestamp 
                        FROM memories 
                        ORDER BY timestamp DESC 
                        LIMIT 3
                    """)
                    recent_memories = c.fetchall()
                    st.write("Summary of recent memories:")
                    for memory in recent_memories:
                        st.write(f"[{memory[1]} - {memory[2]}] {memory[0]}")
                
                elif "forget that" in question.lower():
                    c = conn.cursor()
                    c.execute("DELETE FROM memories WHERE rowid = (SELECT MAX(rowid) FROM memories)")
                    conn.commit()
                    st.write("The last remembered entry has been deleted from the database.")

        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    main()
