# RAG QA Bot with Interactive Interface

This project implements a **Retrieval-Augmented Generation (RAG)** model for a Question Answering (QA) bot that retrieves relevant information from a dataset and generates coherent responses. It utilizes **Pinecone DB** for efficient embedding storage and retrieval, along with the **Cohere API** for generating natural language answers. Additionally, an interactive **Streamlit** interface allows users to upload documents and ask questions based on the content of the documents in real-time.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Example Queries](#example-queries)
- [Challenges and Solutions](#challenges-and-solutions)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project is divided into two main tasks:

- **Part 1**: Development of the RAG-based model to retrieve document embeddings and generate answers based on queries.
- **Part 2**: Creation of an interactive frontend interface where users can upload PDFs, ask questions, and view both the relevant document chunks and generated answers.

The system is containerized using **Docker** for easy deployment and scalability. A **Colab notebook** is provided to demonstrate the pipeline for testing and refinement.

## Features

- Efficient document chunking and embedding generation using **Pinecone DB**.
- Real-time query answering with coherent responses using **Cohere API**.
- Interactive **Streamlit** interface for document uploads and query inputs.
- Scalable and modular architecture with clean backend and frontend separation.
- Supports large documents and multiple queries.
- Dockerized for seamless deployment.

## Architecture

The architecture follows a modular design where:
- **Document Embeddings** are generated and stored in **Pinecone** for retrieval.
- **Queries** are embedded and matched with document chunks to retrieve relevant context.
- **Cohere** API is used to generate natural language answers based on the context and the user's query.

### Process Flow:
1. **Document Upload**: The user uploads a PDF document.
2. **Document Parsing**: The text is extracted from the PDF and split into chunks.
3. **Embeddings Generation**: Document chunks are converted into embeddings using the **Sentence-BERT** model.
4. **Querying**: The user asks a question, and the query is embedded and matched with relevant document chunks from Pinecone.
5. **Answer Generation**: The **Cohere API** generates a natural language response based on the retrieved document chunks.

## Technologies Used

- **Python** for backend processing.
- **Streamlit** for the frontend interface.
- **Pinecone** for vector embedding storage and retrieval.
- **Cohere API** for natural language generation.
- **Sentence-BERT** for document embedding.
- **Docker** for containerization and deployment.
- **PyPDF2** for PDF text extraction.

## Setup and Installation

### Prerequisites
- Docker installed on your system.
- Pinecone API Key.
- Cohere API Key.

### Step-by-Step Setup

1. **Clone the repository**:
   ```bash
   gh repo clone kardam00/QA-Bot-Retrieval-Augmented-Generation
   cd app
   ```

2. **Set up environment variables** for Pinecone and Cohere API keys:
   Create a `.env` file in the project root:
   ```plaintext
   PINECONE_API_KEY=your-pinecone-api-key
   COHERE_API_KEY=your-cohere-api-key
   ```

3. **Build Docker Image**:
   ```bash
   docker build -t qa-bot-app .
   ```

4. **Run the Docker Container**:
   ```bash
   docker run -p 8501:8501 qa-bot-app
   ```

5. **Access the Application**:
   Open your browser and navigate to `http://localhost:8501` to interact with the QA bot.

## Usage

1. **Upload Document**: Upload a PDF document from which the QA bot can retrieve information.
2. **Ask a Question**: Input a question in the text box to retrieve relevant sections of the document and a generated answer.
3. **View Results**: The retrieved document sections and the generated answer will be displayed.

## File Structure

```
RAG-QA-Bot-with-Interactive-UI/
│
├── app/
│   ├── app.py             # Streamlit application code
│   ├── Dockerfile         # Docker configuration for containerizing the app
│   ├── requirements.txt   # Python dependencies
│
├── RAG_model task1.ipynb   # Colab notebook for RAG-based QA model implementation
├── README.md              # Documentation
└── .env.example           # Example environment variables file
```

## Example Queries

After uploading a document, here are some example queries you can use to test the QA bot:
- "What is the main topic discussed in this document?"
- "Summarize the key points from section 3."
- "What are the main challenges mentioned in this report?"

## Challenges and Solutions

- **Handling Large Documents**: Implemented document chunking with optimized embedding generation to handle large documents without memory overflow.
- **Efficient Querying**: Pinecone's vector similarity search allows quick and efficient retrieval of relevant document chunks.
- **Natural Language Generation**: Fine-tuned query context generation to ensure accurate and contextually relevant answers using the Cohere API.

## Contributing

If you'd like to contribute to this project, feel free to open an issue or submit a pull request. Please follow the project's code of conduct.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

By following this guide, you should be able to run, test, and extend the QA bot efficiently.
