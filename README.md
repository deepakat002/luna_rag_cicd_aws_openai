# luna_rag_cicd_aws_openai

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

    M --> N
    style A fill:#e1f5fe
    style M fill:#c8e6c9
    style AA fill:#fff3e0
    style P fill:#f3e5f5