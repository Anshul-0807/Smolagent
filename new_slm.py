from smolagents import CodeAgent, LiteLLMModel, tool, GradioUI
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from typing import Optional, Dict

# Load API keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Define SLM Models using Hugging Face
model_query = LiteLLMModel(
    model_id="microsoft/phi-2",  # Using Phi-2 for general queries
    api_key=HUGGINGFACE_API_KEY,
    model_provider="huggingface"
)

model_code = LiteLLMModel(
    model_id="codellama/CodeLlama-7b-hf",  # Using CodeLlama for code-related tasks
    api_key=HUGGINGFACE_API_KEY,
    model_provider="huggingface"
)

# Global dataset variable
DATASET = None
DATASET_PATH = "dataset/raw_materials_inventory.csv"
# DATABASE_PATH = " "

@tool
def load_dataset_from_csv() -> str:
    """
    Loads the dataset from a CSV file.
    """
    global DATASET
    try:
        DATASET = pd.read_csv(DATASET_PATH)
        return f"Dataset loaded successfully from {DATASET_PATH}. Columns: {', '.join(DATASET.columns)}"
    except Exception as e:
        return f"Error loading dataset from CSV: {str(e)}"

@tool
def load_dataset_from_db() -> str:
    """
    Loads the dataset from a SQLite database.
    """
    global DATASET
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        query = "SELECT * FROM raw_materials_inventory;"
        DATASET = pd.read_sql_query(query, conn)
        conn.close()
        return f"Dataset loaded from database. Columns: {', '.join(DATASET.columns)}"
    except Exception as e:
        return f"Error loading dataset from database: {str(e)}"

@tool
def query_dataset(query: str, data_source: str = "CSV", context: Optional[str] = None) -> str:
    """
    Answers queries about the loaded dataset using Phi-2.
    
    Args:
        query: The user's question about the dataset
        data_source: Source of data ("CSV" or "Database")
        context: Additional context or constraints for the query
    """
    # First load the data from specified source
    if data_source.upper() == "DATABASE":
        load_result = load_dataset_from_db()
    else:
        load_result = load_dataset_from_csv()
    
    if DATASET is None:
        return "Dataset not loaded. Please check the file path and permissions."
    
    try:
        dataset_info = f"""
        Dataset Info:
        - Columns: {', '.join(DATASET.columns)}
        - Sample Data:
        {DATASET.head(3).to_string()}
        - Summary Statistics:
        {DATASET.describe().to_string()}
        """
        
        prompt = f"""Based on the following dataset:
        {dataset_info}
        
        Context: {context if context else 'None'}
        
        Please analyze and answer this question: {query}
        Provide a clear, data-driven response based on the information provided.
        """
        
        response = model_query.complete(prompt)
        return f"{load_result}\n\nAnalysis:\n{response.content}"
        
    except Exception as e:
        return f"Error processing query: {str(e)}"

@tool
def generate_plot(query: str, data_source: str = "CSV") -> str:
    """
    Generates a plot based on the user's query about the dataset.
    
    Args:
        query: The user's request for a specific plot
        data_source: Source of data ("CSV" or "Database")
    """
    # First load the data from specified source
    if data_source.upper() == "DATABASE":
        load_result = load_dataset_from_db()
    else:
        load_result = load_dataset_from_csv()
    
    if DATASET is None:
        return "Dataset not loaded. Please check the file path and permissions."
    
    try:
        plot_prompt = f"""
        Given the query "{query}" and columns {list(DATASET.columns)},
        determine the best type of plot to visualize the data.
        Response should be one of: histogram, scatter, line, bar, box
        """
        plot_type_response = model_code.complete(plot_prompt)
        plot_type = plot_type_response.content.strip().lower()
        
        column_name = query.split()[-1]
        if column_name in DATASET.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if plot_type == "histogram":
                sns.histplot(data=DATASET, x=column_name, kde=True, ax=ax)
            elif plot_type == "scatter":
                sns.scatterplot(data=DATASET, x=DATASET.index, y=column_name, ax=ax)
            elif plot_type == "line":
                sns.lineplot(data=DATASET, x=DATASET.index, y=column_name, ax=ax)
            elif plot_type == "bar":
                sns.barplot(data=DATASET, x=DATASET.index, y=column_name, ax=ax)
            elif plot_type == "box":
                sns.boxplot(data=DATASET, y=column_name, ax=ax)
            
            plt.title(f'Analysis of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Value')
            
            # Save plot to a temporary file
            plt.savefig('temp_plot.png')
            plt.close()
            
            return f"{load_result}\n\nPlot generated successfully for {column_name} using {plot_type} chart type."
        else:
            return f"Column {column_name} not found in dataset."
    except Exception as e:
        return f"Error generating plot: {str(e)}"

@tool
def get_dataset_info(data_source: str = "CSV") -> str:
    """
    Returns information about the currently loaded dataset.
    
    Args:
        data_source: Source of data ("CSV" or "Database")
    """
    # First load the data from specified source
    if data_source.upper() == "DATABASE":
        load_result = load_dataset_from_db()
    else:
        load_result = load_dataset_from_csv()
    
    if DATASET is None:
        return "Dataset not loaded. Please check the file path and permissions."
    
    try:
        info = f"""
        {load_result}
        
        Additional Dataset Information:
        - Number of rows: {len(DATASET)}
        - Number of columns: {len(DATASET.columns)}
        - Data types:\n{DATASET.dtypes.to_string()}
        - Memory usage: {DATASET.memory_usage().sum() / 1024 / 1024:.2f} MB
        """
        return info
    except Exception as e:
        return f"Error getting dataset info: {str(e)}"

# Define Agents for different tasks
agent_csv = CodeAgent(tools=[load_dataset_from_csv], model=model_query)
agent_db = CodeAgent(tools=[load_dataset_from_db], model=model_query)
agent_query = CodeAgent(tools=[query_dataset], model=model_query)
agent_plot = CodeAgent(tools=[generate_plot], model=model_code)
agent_info = CodeAgent(tools=[get_dataset_info], model=model_query)

# Create a main agent that combines all tools
main_agent = CodeAgent(
    tools=[query_dataset, generate_plot, get_dataset_info],
    model=model_query
)

def main():
    # Launch the UI with the main agent
    ui = GradioUI(main_agent)
    ui.launch()

if __name__ == "__main__":
    main()