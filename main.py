import os
import json
import logging
import re
from mistralai import Mistral
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import NLTKTextSplitter

#logging
logging.basicConfig(level=logging.INFO)

#Client 
api_key = "jOx6zygUOixHUCkjnuXlKUmEj7pRYQmN"  # Replace with your Mistral API key
client = Mistral(api_key=api_key)

def call_mistral_api(texts):
    """
    Here, we require this function to return embeddings using Mistral's embed API. 
    Input: txt fiiles
    Returns: list of floats (embeddings)
    """
    try:
        model = "mistral-embed"
        embeddings_response = client.embeddings.create(
            model=model,
            inputs=texts  # Accepts a list of text inputs
        )
        # Extract embeddings from the response
        embeddings = [data.embedding for data in embeddings_response.data]
        return embeddings
    except Exception as e:
        raise Exception(f"Error fetching embeddings from Mistral API: {str(e)}")

class MistralEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return call_mistral_api(texts)

    def embed_query(self, text):
        return call_mistral_api([text])[0]  # Return the embedding for a single query

# Paths
DATASET_DIR = "C:/Users/shifr/OneDrive/Desktop/langchain-assignment-deepstack/langchain-assignment-dataset/stories"
VECTOR_DB_PATH = "./vector_db"


def extract_summary(character_name, character_text):
    # Use a text splitter to break down the text into sentences
    splitter = NLTKTextSplitter()
    sentences = splitter.split_text(character_text)
    
    # Filter sentences that mention the character by name
    relevant_sentences = [s for s in sentences if character_name.lower() in s.lower()]
    
    # Combine the top relevant sentences into a summary
    summary = " ".join(relevant_sentences[:3])  # Select the top 3 sentences for the summary
    return summary



# Compute Embeddings Command
def compute_embeddings():
    logging.info("Starting embeddings computation...")

    if not os.path.exists(DATASET_DIR):
        logging.error(f"Dataset directory '{DATASET_DIR}' not found!")
        return

    loader = DirectoryLoader(DATASET_DIR, glob="*.txt")
    raw_documents = loader.load()

    if not raw_documents:
        logging.error("No documents found in the dataset directory.")
        return

    # wrap documents with real titles from the first line
    documents = []
    for i, doc in enumerate(raw_documents):
        lines = doc.page_content.split('\n')
        title = lines[0].strip()  # first line 
        content = '\n'.join(lines[1:]).strip()  # rest of the content

        documents.append(Document(page_content=content, metadata={"title": title}))

    # Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    logging.info(f"Loaded and split {len(documents)} documents into {len(chunks)} chunks.")

    # Generate embeddings and save them
    embeddings = MistralEmbeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(VECTOR_DB_PATH)
    logging.info(f"Embeddings saved to {VECTOR_DB_PATH}.")


def determine_character_type(character_name, character_text):
    role_keywords = {
        "Protagonist": [
            "main character", "hero", "central figure", "drives the plot", "focus of the story", "important role",
            "key to the story", "saves the day", "leader", "fights against evil", "overcomes obstacles", "quest",
            "journey", "sacrifice", "protector", "champion", "redeems", "chosen one", "guiding force", 
            "narrative centers around", "rises to the challenge", "determined", "brave", "selfless", "courageous"
        ],
        "Antagonist": [
            "villain", "enemy", "opposes", "creates obstacles", "threat", "adversary", "destructive", "rival",
            "forces conflict", "evil", "dark", "manipulative", "antagonizes", "power-hungry", "self-serving",
            "sinister", "corrupt", "foil", "desires power", "causes harm", "bad", "despised", "destructive force"
        ],
        "Supporting": [
            "helps", "companion", "sidekick", "mentor", "trusted ally", "adviser", "helper", "aids the protagonist",
            "secondary character", "assists", "guides", "provides support", "assistant", "friend", "team member", 
            "protector", "encourages", "loyal", "counselor"
        ],
        "Neutral": [
            "observer", "bystander", "uninvolved", "neutral role", "does not take sides", "indifferent", "ambiguous",
            "apathetic", "no direct involvement", "on the sidelines", "in the background", "passive", "uninfluenced"
        ],
        "Antihero": [
            "flawed hero", "morally ambiguous", "grey morality", "seeks redemption", "nontraditional hero", 
            "struggles with morality", "questionable methods", "outsider", "reluctant hero", "does the right thing, but",
            "unpredictable", "doesn’t follow traditional hero rules", "selfish but just", "tough but compassionate"
        ]
    }

    logging.info(f"Analyzing character: {character_name}")
    for role, keywords in role_keywords.items():
        for keyword in keywords:
            if keyword.lower() in character_text.lower():
                logging.info(f"Matched keyword '{keyword}' for role '{role}'")
                return role

    logging.warning(f"No matching keywords found for character '{character_name}'. Returning 'Unknown'.")
    return "Unknown"


def extract_relations(character_name, character_text):

    relations = []

    patterns = [
        (r"(\bfather\b|\bfathered\b)", "Father"),
        (r"(\bmother\b|\bmothered\b)", "Mother"),
        (r"(\bsibling\b|\bbrother\b|\bsister\b)", "Sibling"),
        (r"(\bfriends?\b|\bcompanion\b|\bpartner\b)", "Companion"),
        (r"(\bson\b|\bdaughter\b)", "Child"),
        (r"(\bhusband\b|\bwife\b)", "Spouse"),
        (r"(\bmentor\b|\bguide\b)", "Mentor"),
    ]

    for pattern, relation_type in patterns:
        matches = re.findall(pattern, character_text, re.IGNORECASE)
        if matches:
            for match in matches:
                relations.append({
                    "name": match,
                    "relation": relation_type
                })
    
    return relations


def get_character_info(character_name):
    logging.info(f"Fetching information for character: {character_name}")

    if not os.path.exists(VECTOR_DB_PATH):
        logging.error("Vector database not found. Please run `compute-embeddings` first.")
        return

    embeddings = MistralEmbeddings()
    vector_db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    query = f"Find details about the character {character_name}."
    results = vector_db.similarity_search(query, k=5)

    if not results:
        logging.warning(f"Character '{character_name}' not found.")
        return

    raw_summary = results[0].page_content[:250]
    clean_summary = " ".join(raw_summary.split("\n")).strip()

    character_text = results[0].page_content 
    relations = extract_relations(character_name, character_text)  

    structured_data = {
        "name": character_name,
        "storyTitle": results[0].metadata.get("title", "Unknown"),  
        "summary": clean_summary,
        "relations": relations, 
        "characterType": determine_character_type(character_name, results[0].page_content), 
    }

    print(json.dumps(structured_data, indent=4))



def main():
    import argparse

    parser = argparse.ArgumentParser(description="LangChain Assignment Commands")
    subparsers = parser.add_subparsers(dest="command")

    # Compute embeddings command
    parser_compute = subparsers.add_parser("compute-embeddings", help="Compute embeddings for stories")

    # Get character info command
    parser_info = subparsers.add_parser("get-character-info", help="Get character information")
    parser_info.add_argument("character_name", type=str, help="Name of the character")

    args = parser.parse_args()

    if args.command == "compute-embeddings":
        compute_embeddings()
    elif args.command == "get-character-info":
        get_character_info(args.character_name)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
