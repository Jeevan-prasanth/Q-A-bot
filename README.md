# Q-A-bot

# NVIDIA CUDA Documentation Search

This project involves web scraping, data chunking and embedding, vector database creation, and a web application that allows users to search for information using a hybrid retrieval system. The final application runs using Streamlit.

## Project Structure

- `webscraping.py`: Script to scrape data from the web.
- `chunk-embed.py`: Script to chunk and embed the scraped data.
- `vector-db.py`: Script to create and populate the vector database.
- `app.py`: The main application script that runs the Streamlit app.

## Setup Instructions

### 1. Clone the Repository

```sh
git clone https://github.com/Jeevan-prasanth/Q-A-bot.git
cd your-repo
```
### 2. Create and Activate a Virtual Environment
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
### 3. Install the Dependencies
```sh
pip install -r requirements.txt
```


## Running the Project

### 1. Run Web Scraping
First, run the web scraping script to gather data:
```sh
python webscraping.py
```
## 2. Run Data Chunking and Embedding
Next, run the script to chunk and embed the scraped data:
```sh
python chunk-embed.py
```
## 3. Create and Populate the Vector Database
Then, run the script to create and populate the vector database:
```sh
python vector-db.py
```
## 4. Run the Application
Finally, run the Streamlit application:
```sh
streamlit run app.py
```

## Environment Variables
Make sure to set up the necessary environment variables. You can create a .env file in the project root directory with the following content:
```sh
GOOGLE_API_KEY=your_google_api_key
```
### Get token and cluster endpoint from Zilliz Cloud and place it in the vector-db.py and app.py
