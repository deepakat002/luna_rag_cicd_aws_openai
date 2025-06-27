# Luna 🐶 - Dog Expert RAG System

Luna is designed to be your friendly, knowledgeable companion for all things dog-related. Whether you're a new dog owner or an experienced handler, Luna provides expert guidance on:

1. Breed Information: Characteristics, temperament, and care requirements
2. Training Advice: Behavioral guidance and training techniques
3. Health & Nutrition: Feeding guidelines and health monitoring
4. General Care: Daily care routines and best practices
---

## ⚙️ How It Works

1. **PDF Ingestion**  
   Loads dog-related PDFs, splits them into text chunks, and stores them as vector embeddings in ChromaDB.

2. **Retriever**  
   Uses the ChromaDB vector store to find relevant chunks for any user question.

3. **Chat Memory**  
   Persists previous conversation messages (using ConversationBufferWindowMemory) to maintain context across turns.

4. **Prompt**  
   Feeds both retrieved document chunks and chat history into a carefully crafted prompt for the OpenAI chat model.

5. **Greeting Handler**  
   Detects and handles simple greetings separately from knowledge queries, providing a friendly greeting instantly.

6. **Response**  
   Generates a concise answer (≤50 words), referencing the retrieved context whenever relevant.

7. **Persistence**  
   Chat history is stored to the filesystem so users can resume conversations.

---

## 📐 System Architecture:

```mermaid
graph TB
    subgraph "Initialization Phase"
        A[Start Luna System] --> B[Check OpenAI API Key]
        B --> C[Setup Directories]
        C --> D[Check for Existing Vector Store]
        D --> E{Vector Store Exists?}
        E -->|Yes| F[Load Existing ChromaDB]
        E -->|No| G[Load PDFs from Directory]
        G --> H[Split Documents into Chunks]
        H --> I[Create Embeddings]
        I --> J[Store in ChromaDB]
        F --> K[Setup Retriever]
        J --> K
        K --> L[Initialize LLM Chain]
        L --> M[System Ready]
    end

    subgraph "Chat Processing Flow"
        N[User Message] --> O[Create/Get Session ID]
        O --> P{Is Greeting?}
        P -->|Yes| Q[Generate Greeting Response]
        P -->|No| R[Show Thinking Indicator]
        Q --> S[Save to History]
        R --> T[Vector Search]
        T --> U[Retrieve Relevant Documents]
        U --> V[Construct Prompt with Context]
        V --> W[Send to OpenAI LLM]
        W --> X[Generate Response]
        X --> Y[Determine Response Source]
        Y --> Z[Save to History]
        S --> AA[Send Response to User]
        Z --> AA
        AA --> AB[Update Memory]
        AB --> AC[Ready for Next Message]
    end

    subgraph "Data Management"
        AD[PDF Documents] --> AE[DirectoryLoader]
        AE --> AF[Document Chunks]
        AF --> AG[OpenAI Embeddings]
        AG --> AH[ChromaDB Vector Store]
        AI[Chat History] --> AJ[Session Files]
        AJ --> AK[Conversation Memory]
    end

