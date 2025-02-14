# FIRST CODEBASE

from smolagents import CodeAgent, LiteLLMModel, tool, GradioUI
import openai
import os
import pandas as pd
from typing import Optional, Dict

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global variables
DATASET = None
DATASET_PATH = "dataset/raw_materials_inventory.csv"  # Predefined dataset path

# Step 1: Define the LLM Model (GPT-4)
model = LiteLLMModel(model_id="gpt-4", api_key=OPENAI_API_KEY)

def initialize_dataset():
    """Initialize the dataset when the program starts"""
    global DATASET
    try:
        DATASET = pd.read_csv(DATASET_PATH)
        print(f"Dataset loaded successfully from {DATASET_PATH}")
        print(f"Columns: {', '.join(DATASET.columns)}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")

@tool
def query_dataset(query: str, context: Optional[str] = None) -> str:
    """
    Answers queries about the loaded dataset.

    Args:
        query: The user's question about the dataset
        context: Additional context or constraints for the query

    Returns:
        A response based on the dataset content

    Example:
        answer = query_dataset("What is the average sales value?")
    """
    global DATASET
    
    if DATASET is None:
        return "Dataset not loaded. Please check the file path and permissions."
    
    try:
        # Create a context string with dataset information
        dataset_info = f"""
        Dataset Information:
        - Columns: {', '.join(DATASET.columns)}
        - Data sample: {DATASET.head(3).to_string()}
        - Basic stats: {DATASET.describe().to_string()}
        """
        
        # Create the prompt for GPT-4
        prompt = f"""Based on the following dataset information:
        {dataset_info}
        
        Additional context: {context if context else 'None provided'}
        
        Please answer this question: {query}
        
        Provide a clear, accurate answer based only on the data provided.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a data analysis expert. Analyze the provided dataset and answer questions accurately based solely on the given data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more focused answers
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing query: {str(e)}"

@tool
def get_dataset_info() -> str:
    """
    Returns information about the currently loaded dataset.

    Args:
        None

    Returns:
        A string containing information about the dataset

    Example:
        info = get_dataset_info()
    """
    global DATASET
    
    if DATASET is None:
        return "Dataset not loaded. Please check the file path and permissions."
    
    try:
        info = f"""
        Dataset Information:
        - Number of rows: {len(DATASET)}
        - Number of columns: {len(DATASET.columns)}
        - Columns: {', '.join(DATASET.columns)}
        - Data types:\n{DATASET.dtypes.to_string()}
        - Memory usage: {DATASET.memory_usage().sum() / 1024 / 1024:.2f} MB
        """
        return info
    except Exception as e:
        return f"Error getting dataset info: {str(e)}"

# Create the Agent with the new tools
agent = CodeAgent(
    tools=[query_dataset, get_dataset_info],
    model=model
)

# Create and launch the Gradio UI
def main():
    # Initialize dataset when program starts
    initialize_dataset()
    # Launch the UI
    ui = GradioUI(agent)
    ui.launch()

if __name__ == "__main__":
    main()