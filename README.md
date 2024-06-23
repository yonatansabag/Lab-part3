# **RAG Pipeline with Pinecone VectorDB and Cohere**

This repository contains an implementation of a basic Retrieval-Augmented Generation (RAG) pipeline using Pinecone VectorDB and the Cohere API. The goal is to enhance the performance of a standard Question Answering (QA) model by leveraging relevant documents from a vector database.

## **Dataset**

The dataset used for this project is the `Ateeqq/news-title-generator` dataset from Hugging Face. The dataset contains text fields which are used for embedding and retrieval.

## **Requirements**

- Python 3.6+
- `sentence-transformers`
- `datasets`
- `pinecone-client`
- `cohere`
- `tqdm`
- `numpy`

Install the required libraries using pip:

```sh
pip install sentence-transformers datasets pinecone-client cohere tqdm numpy
```
## **Usage**

Step 1: Load and Embed the Dataset
Load the dataset and embed the text fields using a sentence-transformer model with chunking to handle longer documents.

Step 2: Upsert Embeddings to Pinecone
Upsert the generated embeddings to the Pinecone index for efficient retrieval of relevant documents.

Step 3: Query the Model
Use the Cohere API to query the model with an augmented prompt that includes retrieved contexts from the Pinecone index.

Step 4: Display the Response
Display the model's response in a nicely formatted HTML format for better readability in a Colab notebook.

## **Project Structure**

Lab_hw1_part3.ipynb: The main Colab notebook containing the implementation of the RAG pipeline.
README.md: This file, providing an overview of the project and instructions for usage.

## **Acknowledgements**
Sentence Transformers

Hugging Face Datasets

Pinecone Vector Database

Cohere API

## **License**

This project is licensed under the MIT License.
