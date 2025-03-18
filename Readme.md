# News Summarization

## Overview
This project is a **News Sentiment Analysis and Summarization** tool built using **Streamlit**, **Natural Language Processing (NLP)**, and **Web Scraping**. It extracts news articles, summarizes their content, performs sentiment analysis, and provides insights into media perception about a company or topic.

## Features
- **Fetch News Articles**: Retrieves relevant news articles from Bing News.
- **Text Extraction**: Extracts clean text from news articles.
- **Summarization**: Uses LLM-based summarization to condense article content.
- **Sentiment Analysis**: Determines whether articles have a positive, negative, or neutral tone.
- **Topic Extraction**: Identifies key topics from the articles.
- **Comparative Analysis**: Compares sentiment and topics across multiple articles.
- **Multilingual Support**: Translates summaries into Hindi with text-to-speech functionality.
- **Interactive Query Processing**: Allows users to ask questions about the news data.

## Tech Stack
- **Python**
- **Streamlit** (for UI)
- **Newspaper3k** (for web scraping)
- **BeautifulSoup** (for additional scraping support)
- **spaCy** (for NLP tasks)
- **TextBlob** (for sentiment analysis)
- **LangChain** (for LLM-based processing)
- **FAISS** (for efficient text retrieval)
- **GoogleTranslator** (for translations)
- **gTTS** (for text-to-speech in Hindi)

## Installation
### Prerequisites
Ensure you have Python 3.10 installed.

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/news-sentiment-analysis.git
   cd news-sentiment-analysis
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up API keys**:
   - Create a `.env` file in the root directory.
   - Add your `GROQ_API_KEY`:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

## Usage
Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

### Steps to Use
1. Enter a **company name** or keyword to fetch news articles.
2. The system will:
   - Scrape news articles
   - Extract and summarize text
   - Perform sentiment analysis
   - Extract key topics
   - Generate a comparative analysis
3. View **summaries, sentiment breakdown, and key insights**.
4. Enter a **custom query** to analyze extracted data interactively.
5. Play **Hindi audio translation** of the sentiment summary.

## File Structure
```
news-sentiment-analysis/
│── streamlit_app.py        # Main Streamlit UI
│── requirements.txt        # List of dependencies
│── README.md               # Project documentation
```

## Future Enhancements
- Add support for more languages.
- Improve sentiment analysis with deep learning models.
- Expand to other news sources.
- Implement a more interactive chatbot.

## License
This project is licensed under the MIT License.

---



