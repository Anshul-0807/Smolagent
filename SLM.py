from smolagents import CodeAgent, LiteLLMModel, tool
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from typing import Optional
import io
import base64
from PIL import Image

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

# Paths
DATASET_PATH = "dataset/raw_materials_inventory.csv"
# DATABASE_PATH = " "

@tool
def load_dataset_from_csv() -> str:
    """
    Loads the dataset from a CSV file.

    Returns:
        str: A message indicating the success or failure of loading the dataset

    Example:
        result = load_dataset_from_csv()
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

    Returns:
        str: A message indicating the success or failure of loading the dataset

    Example:
        result = load_dataset_from_db()
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
def query_dataset(query: str, context: Optional[str] = None) -> str:
    """
    Answers queries about the loaded dataset using Phi-2.

    Args:
        query: The user's question or query about the dataset
        context: Additional context or constraints for the query (optional)

    Returns:
        str: A response based on the dataset content

    Example:
        answer = query_dataset("What is the average inventory value?", "Focus on last month")
    """
    global DATASET
    if DATASET is None:
        return "Dataset not loaded. Please load the dataset first."
    
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
        
        # Using Phi-2 for analysis
        response = model_query.complete(prompt)
        return response.content
        
    except Exception as e:
        return f"Error processing query: {str(e)}"

@tool
def generate_plot(query: str) -> plt.Figure:
    """
    Generates a plot based on the user's query about the dataset.

    Args:
        query: The user's request for a specific plot or visualization

    Returns:
        plt.Figure: The generated matplotlib figure

    Example:
        plot = generate_plot("Create a plot for inventory levels")
    """
    global DATASET
    if DATASET is None:
        return None
    
    try:
        # Use CodeLlama to determine the best plot type
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
            return fig
        else:
            return None
    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")
        return None

# Define Agents
agent_csv = CodeAgent(tools=[load_dataset_from_csv], model=model_query)
agent_db = CodeAgent(tools=[load_dataset_from_db], model=model_query)
agent_query = CodeAgent(tools=[query_dataset], model=model_query)
agent_plot = CodeAgent(tools=[generate_plot], model=model_code)

class MultiAgentCoordinator:
    def __init__(self):
        self.agent_csv = agent_csv
        self.agent_db = agent_db
        self.agent_query = agent_query
        self.agent_plot = agent_plot

    def process_query(self, task: str, use_db: bool = False):
        """
        Process user query and return appropriate response
        """
        try:
            # Load data
            if use_db:
                load_response = self.agent_db.run("load_dataset_from_db")
            else:
                load_response = self.agent_csv.run("load_dataset_from_csv")
            st.info(load_response)
            
            # Handle query based on type
            if "plot" in task.lower() or "chart" in task.lower():
                result = self.agent_plot.run("generate_plot", query=task)
                if result is not None:
                    st.pyplot(result)
                else:
                    st.warning("Could not generate plot for the given query.")
            else:
                result = self.agent_query.run("query_dataset", query=task)
                st.write(result)
                
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

def main():
    st.set_page_config(page_title="Data Analysis Assistant", layout="wide")
    
    st.title("Data Analysis Assistant")
    st.markdown("*Powered by Phi-2 and CodeLlama*")
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    data_source = st.sidebar.radio(
        "Select Data Source",
        ("CSV File", "Database")
    )
    
    # Main query input
    user_query = st.text_area("Enter your query:", height=100)
    
    # Process button
    if st.button("Process Query"):
        if user_query:
            with st.spinner("Processing your query..."):
                coordinator = MultiAgentCoordinator()
                use_db = data_source == "Database"
                coordinator.process_query(user_query, use_db)
        else:
            st.warning("Please enter a query first.")
    
    # Display dataset if loaded
    if DATASET is not None:
        st.sidebar.markdown("### Dataset Preview")
        st.sidebar.dataframe(DATASET.head())
        
        st.sidebar.markdown("### Dataset Info")
        st.sidebar.text(f"Rows: {len(DATASET)}")
        st.sidebar.text(f"Columns: {len(DATASET.columns)}")

if __name__ == "__main__":
    main()