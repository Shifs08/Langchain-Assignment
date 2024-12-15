# Langchain-Assignment

This project uses the power of **LangChain**, **Mistral AI**, and **FAISS** for text embedding and similarity search to analyze character data from stories. This README helps you to get started! 

---

## Features 🌟

- **Character Analysis**: Identify relationships, roles, and summaries of characters from story texts.
- **Text Embedding**: Generates embeddings using Mistral's `embed` API for similarity search.
- **Document Management**: Supports directory loading and document chunking for efficient processing.

---

## Setup 

### Prerequisites 
- Python 3.7+
- `pip` for managing Python packages

### Installation 
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your **Mistral API Key** to the `main.py` file:
   ```python
   api_key = "your_api_key_here"
   ```

---

## Usage 🛠

### 1. Compute Embeddings 
Generate embeddings from story documents in the specified dataset directory:
```bash
python main.py compute-embeddings
```
- **Dataset Path**: Update `DATASET_DIR` in the code with your story files directory.

### 2. Analyze a Character 
Get information about a character:
```bash
python main.py get-character-info "Character Name"
```
- Replace `"Character Name"` with the name of the character you'd like to analyze.

### Example Outputs 
Output structure:
```json
{
    "name": "Character Name",
    "storyTitle": "Story Title",
    "summary": "Character summary here...",
    "relations": [
        { "name": "Relation Name", "relation": "Relation Type" }
    ],
    "characterType": "Protagonist/Antagonist/..."
}
```


---

## File Structure 

```
.
├── main.py                 # Main script for running commands
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── vector_db/              # Directory to store FAISS vector database
├── dataset/                # Directory for input story files
```

---


## Acknowledgments 
- **[LangChain](https://github.com/hwchase17/langchain)** for providing modular AI building blocks.
- **[Mistral](https://www.mistral.ai/)** for advanced embedding APIs.
- **[FAISS](https://github.com/facebookresearch/faiss)** for efficient similarity search.

---
