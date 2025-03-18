import streamlit as st
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import time
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.schema import Document
from textblob import TextBlob  
from deep_translator import GoogleTranslator
from gtts import gTTS
from urllib.parse import urljoin
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

# Load environment variables

api_key = "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"

# Load NLP model for topic extraction
nlp = spacy.load("en_core_web_sm")

# Initialize LLM
llm = ChatGroq(groq_api_key=api_key, model="Gemma2-9b-It")

def fetch_news_links_bing(company, max_articles=10):
    """Fetch at least 5 unique news article links from Bing News"""
    search_url = f"https://www.bing.com/news/search?q={company.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    news_links = set()
    base_url = "https://www.bing.com"

    for link in soup.find_all("a", href=True):
        url = urljoin(base_url, link["href"])  # Convert to absolute URL
        if url.startswith("http") and "bing.com" not in url and url not in news_links:
            news_links.add(url)
        if len(news_links) >= max_articles:
            break

    if len(news_links) < max_articles:
        st.warning(f"Warning: Only {len(news_links)} unique articles found.")
    
    return list(news_links)[:max_articles]

def extract_article_text(url):
    """Extract clean article text using newspaper3k"""
    try:
        article = Article(url)
        article.download()
        article.parse()

        if not article.text.strip():
            return "Error extracting content", "No content extracted."

        return article.title, article.text
    except Exception as e:
        return "Error extracting content", f"Error extracting content: {str(e)}"

def process_query(texts, query):
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
    split_texts = text_splitter.split_text(texts)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(split_texts, embeddings)

    retriever = db.as_retriever()
    model = ChatGroq(model="Gemma2-9b-It", groq_api_key=api_key)

    # Contextualization prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

    # System prompt for answering questions
    system_prompt = (
        "You are an AI assistant that helps summarize and answer questions from documents.\n\n"
        "Context:\n{context}\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User Question:\n{input}"
    )

    qa_prompt = ChatPromptTemplate.from_template(system_prompt)

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    chat_history = []
    response = rag_chain.invoke({"input": query, "chat_history": chat_history})

    return response['answer']

def extract_key_topics(text):
    """Extracts key topics using spaCy NLP"""
    doc = nlp(text)
    topics = {ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "EVENT", "PERSON"]}
    return list(topics)

def summarize_text(article_text):
    """Summarizes the article content using LLM"""
    if not article_text.strip():
        return "Summary not available due to extraction failure."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents([Document(page_content=article_text)])

    prompt_template = PromptTemplate(
        input_variables=['text'],
        template="Summarize the following article:\n{text}"
    )

    summary_chain = LLMChain(llm=llm, prompt=prompt_template)

    summary_output = summary_chain.run({"text": docs[0].page_content})  # Summarizing first chunk
    return summary_output.strip() if summary_output else "Summary not generated."

def analyze_sentiment(text):
    """Performs sentiment analysis using TextBlob"""
    sentiment_score = TextBlob(text).sentiment.polarity
    return "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

def generate_sentiment_summary(articles_data):
    """Generates summary insights based on sentiment analysis"""
    total_articles = len(articles_data)
    
    if total_articles == 0:
        return "No valid articles were processed for analysis."

    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for article in articles_data:
        sentiment_counts[article["Sentiment"]] += 1

    # Calculate percentages
    positive_pct = round((sentiment_counts["Positive"] / total_articles) * 100, 2)
    negative_pct = round((sentiment_counts["Negative"] / total_articles) * 100, 2)
    neutral_pct = round((sentiment_counts["Neutral"] / total_articles) * 100, 2)

    summary = f"""
    Sentiment Analysis Summary for {company_name} News Coverage:
    - Positive Articles: {sentiment_counts['Positive']} ({positive_pct}%)
    - Negative Articles: {sentiment_counts['Negative']} ({negative_pct}%)
    - Neutral Articles: {sentiment_counts['Neutral']} ({neutral_pct}%)

    Insights:
    - The majority of news articles are {'positive' if positive_pct > negative_pct else 'negative' if negative_pct > positive_pct else 'neutral'}.
    - There is {abs(positive_pct - negative_pct)}% difference between positive and negative coverage.
    - This analysis suggests that media perception of {company_name} is {('generally favorable' if positive_pct > negative_pct else 'somewhat critical' if negative_pct > positive_pct else 'balanced')}.
    """
    return summary.strip()
def generate_comparative_analysis(articles_data):
    comparisons = []
    topics = {"Common Topics": set(), "Unique Topics": {}}
    for i, article in enumerate(articles_data):
        topics["Unique Topics"][f"Article {i+1}"] = set(article["Topics"])
    if len(articles_data) > 1:
        for i in range(len(articles_data) - 1):
            comparisons.append({
                "Comparison": f"{articles_data[i]['Title']} vs {articles_data[i+1]['Title']}",
                "Impact": f"{articles_data[i]['Summary']} | {articles_data[i+1]['Summary']}"
            })
    return {"Comparative Analysis": comparisons, "Topic Overlap": topics}

def generate_summary(articles_data):
    """Generates summary insights based on sentiment analysis"""
    total_articles = len(articles_data)
    
    if total_articles == 0:
        return "No valid articles were processed for analysis."

    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for article in articles_data:
        sentiment_counts[article["Sentiment"]] += 1

    # Calculate percentages
    positive_pct = round((sentiment_counts["Positive"] / total_articles) * 100, 2)
    negative_pct = round((sentiment_counts["Negative"] / total_articles) * 100, 2)
    neutral_pct = round((sentiment_counts["Neutral"] / total_articles) * 100, 2)

    summary = f"""
    Sentiment Analysis Summary for {company_name} News Coverage:
    - Positive Articles: {sentiment_counts['Positive']} ({positive_pct}%)
    - Negative Articles: {sentiment_counts['Negative']} ({negative_pct}%)
    - Neutral Articles: {sentiment_counts['Neutral']} ({neutral_pct}%)

    Insights:
    - The majority of news articles are {'positive' if positive_pct > negative_pct else 'negative' if negative_pct > positive_pct else 'neutral'}.
    - There is {abs(positive_pct - negative_pct)}% difference between positive and negative coverage.
    - This analysis suggests that media perception of {company_name} is {('generally favorable' if positive_pct > negative_pct else 'somewhat critical' if negative_pct > positive_pct else 'balanced')}.
    """
    return summary.strip()
# Streamlit UI
st.title("News Sentiment Analysis and Summarization")
company_name = st.text_input("Enter company name:")

        
if company_name:
    # Fetch news articles
    news_links = fetch_news_links_bing(company_name, max_articles=10)

    articles_data = []
    articles_text = []

    if news_links:
        for idx, link in enumerate(news_links):
            st.write(f"\nFetching article [{idx+1}]: {link}")
            time.sleep(1)
            
            title, article_text = extract_article_text(link)
            if article_text.startswith("Error"):
                continue  # Skip faulty articles
            articles_text.append(article_text)
            summary = summarize_text(article_text)
            key_topics = extract_key_topics(summary)
            sentiment = analyze_sentiment(summary)
            
            articles_data.append({
                "Title": title,
                "Summary": summary,
                "Sentiment": sentiment,
                "Topics": key_topics
            })

        sentiment_summary = generate_sentiment_summary(articles_data)
        
        comparative_analysis = generate_comparative_analysis(articles_data)
        summary = generate_summary(articles_data)
        output_data = {
            "Company": company_name,
            "Articles": articles_data,
            "Comparative Sentiment Score": {
                "Sentiment Distribution": sentiment_summary,
                "Coverage Differences": comparative_analysis["Comparative Analysis"],
                "Topic Overlap": comparative_analysis["Topic Overlap"]
            },
            "Final Sentiment Analysis": summary,
            "Audio": "[Play Hindi Speech]"
        }
        
        st.json(output_data)
        hindi_translation = GoogleTranslator(source='en', target='hi').translate(output_data["Final Sentiment Analysis"])
        tts = gTTS(text=hindi_translation, lang='hi')
        tts.save("hindi_audio.mp3")
        st.audio("hindi_audio.mp3")

        
    #
    query = st.text_input("Please enter your query related to the article:")

    if query:
        query_process = process_query(' '.join([article for article in articles_text]), query)
        st.subheader("Query Response")
        st.write(query_process)
