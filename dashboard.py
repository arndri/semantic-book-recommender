import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

books = pd.read_csv('books_with_emotions.csv')
books['img'] = books['thumbnail'] + "&fife=w800"
books['img'] = np.where(books['img'].isnull(), 'cover-not-found.jpg', books['img'])

raw_doc = TextLoader('tagged_desc.txt',encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=0,chunk_overlap=0,separator='\n')
docs = text_splitter.split_documents(raw_doc)
db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

def retrieve_recommendation (
        query: 'str',
        category:'str' = None,
        tone: 'str' = None,
        initial_top_k: 'int' = 50,
        final_top_k: 'int' = 16,
) -> pd.DataFrame:
    recs = db.similarity_search(query, k=initial_top_k)
    list_of_books = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books['isbn13'].isin(list_of_books)].head(final_top_k)

    if category != 'All':
        book_recs = book_recs[book_recs['new_categories'] == category][:final_top_k]
    else:
        book_recs = book_recs[:final_top_k]
    
    if tone == 'Happy':
        book_recs.sort_values(by='joy', ascending=False, inplace=True)
    elif tone == 'Angry':
        book_recs.sort_values(by='anger', ascending=False, inplace=True)
    elif tone == 'Sad':
        book_recs.sort_values(by='sadness', ascending=False, inplace=True)
    elif tone == 'Fear':
        book_recs.sort_values(by='fear', ascending=False, inplace=True)
    elif tone == 'Surprise':
        book_recs.sort_values(by='surprise', ascending=False, inplace=True)

    return book_recs

def recommend_books(query, category, tone):
    recommendations = retrieve_recommendation(query, category, tone)
    result = []

    for _, row in recommendations.iterrows():
        desc = row['description']
        truncated_desc = desc[:30] + '...' if len(desc) > 30 else desc

        authors_split = row['authors'].split(';')
        if len(authors_split) == 2:
            authors = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors = f"{authors_split[0]} et al."
        else:
            authors = authors_split[0]

        caption = f"{row['title']} by {authors}: {truncated_desc}"
        result.append((row['img'], caption))
    return result

categories = ['All'] + sorted(books['new_categories'].unique())
tones = ['All'] + ['Happy', 'Angry', 'Sad', 'Fear', 'Surprise']

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:

    gr.Markdown("Semantic Books Recommender")

    with gr.Row():
        query = gr.Textbox(label="Enter a book title or description", placeholder="e.g. Story about war")
        category = gr.Dropdown(label="Select a category", choices=categories, value='All')
        tone = gr.Dropdown(label="Select a tone", choices=tones, value='All')
        submit = gr.Button("Get Recommendations")

    gr.Markdown("Recommended Books")
    output = gr.Gallery(label = "Boooks", columns=8, rows=2)
    submit.click(fn=recommend_books, inputs=[query, category, tone], outputs=output)


if __name__ == '__main__':
    dashboard.launch()