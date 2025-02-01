![image](https://github.com/user-attachments/assets/c3a80326-9de1-4f67-83b9-4f365bd0ec0b)
# Semantic Book Recommender

## Overview

The Semantic Book Recommender is a machine learning-based application that provides book recommendations based on semantic analysis. It uses advanced natural language processing (NLP) techniques to understand the content and emotions of book descriptions and recommends books that match the user's query, category, and tone preferences.

## Features

- **Semantic Search**: Uses embeddings to find books that are semantically similar to the user's query.
- **Category Filtering**: Allows users to filter recommendations by book categories.
- **Tone Filtering**: Sorts recommendations based on the emotional tone (e.g., Happy, Angry, Sad, Fear).
- **Interactive Interface**: Provides an interactive interface using Gradio for easy user interaction.

## Tools
- **LangChain
- **SentenceTransformers
- **Transformers (Bart for labeling, Distil Roberta for sentiment analysis)
- **Chroma (OpenAI embeddings)
- **Gradio


## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/semantic-book-recommender.git
cd semantic-book-recommender
