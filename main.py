import scrapy
from bs4 import BeautifulSoup
import spacy
import pandas as pd
from datasets import Dataset
from transformers import pipeline
import nltk
from fastapi import FastAPI, HTTPException
import dvc
import mlflow
import logging
from pydantic import BaseModel
import uvicorn
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# Initialize NLP components
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

app = FastAPI()
logging.basicConfig(level=logging.INFO)

class WebsiteSpider(scrapy.Spider):
    name = "website_spider"
    start_urls = []

    def __init__(self, start_url=None, *args, **kwargs):
        super(WebsiteSpider, self).__init__(*args, **kwargs)
        if start_url:
            self.start_urls = [start_url]

    def parse(self, response):
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        page_text = ' '.join([para.get_text() for para in paragraphs])
        self.crawler.stats.get_value('_data').append({'text': page_text})

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

def generate_qa_pairs(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    qa_pairs = []
    for sentence in sentences:
        result = qa_pipeline(question="What is this passage about?", context=sentence)
        qa_pairs.append({"question": result['question'], "answer": result['answer'], "context": sentence})
    return qa_pairs

def create_dataset(data):
    df = pd.DataFrame(data)
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.save_to_disk("generated_dataset")
    return "generated_dataset"

def version_dataset(path):
    dvc.add(path)
    return path

class GenerateDatasetRequest(BaseModel):
    url: str

@app.post("/generate_dataset")
def generate_dataset_endpoint(request: GenerateDatasetRequest):
    return generate_dataset_logic(request.url)

@app.get("/")
def generate_dataset_example():
    return generate_dataset_logic("https://www.earthdefenderstoolkit.com/one-pager/guide-librerouter-and-libremesh/")

def generate_dataset_logic(url: str):
    try:
        mlflow.start_run()

        process = CrawlerProcess(get_project_settings())
        process.crawl(WebsiteSpider, start_url=url)
        process.start()

        # Collect the scraped data
        scraped_data = process.stats.get_value('_data')
        if not scraped_data:
            raise ValueError("No data scraped from the website.")

        full_text = ' '.join([item['text'] for item in scraped_data])
        preprocessed_text = preprocess_text(full_text)
        qa_pairs = generate_qa_pairs(preprocessed_text)
        dataset_path = create_dataset(qa_pairs)
        version_dataset(dataset_path)

        mlflow.end_run()
        return {"status": "Dataset generated successfully", "path": dataset_path}
    except Exception as e:
        mlflow.end_run(status='FAILED')
        logging.error(f"Error generating dataset: {e}")
        raise HTTPException(status_code=500, detail="Dataset generation failed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
