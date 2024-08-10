from langchain_community.document_loaders import YoutubeLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap
import os

os.environ["OPENAI_API_KEY"] = "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"

# Load environment variables
load_dotenv(find_dotenv())

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db, transcript

def summarize_video(transcript):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)

    summary_prompt = """
        You are a helpful assistant that summarizes the content of YouTube video transcripts.
        Please provide a concise summary of the following transcript:
        
        {transcript}
        
        """
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(summary_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    summary = chain.run(transcript=transcript[0].page_content)
    return summary.strip()

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that can answer questions about YouTube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


# Main Program
if __name__ == "__main__":
    # Ask the user for the YouTube video URL
    video_url = input("Please enter the YouTube video URL: ")

    # Create the database and get the transcript
    db, transcript = create_db_from_youtube_video_url(video_url)

    # Summarize the video
    print("Summarizing the video...")
    summary = summarize_video(transcript)
    print("\nVideo Summary:\n")
    print(textwrap.fill(summary, width=50))

    # Allow the user to ask questions about the video
    while True:
        query = input("\nAsk any question about the video (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response, docs = get_response_from_query(db, query)
        print("\nAnswer:\n")
        print(textwrap.fill(response, width=50))
