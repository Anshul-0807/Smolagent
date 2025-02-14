from smolagents import CodeAgent, LiteLLMModel, tool
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from typing import Optional
import openai
import io
import base64
from PIL import Image

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define LLM Models
model_query = LiteLLMModel(
    model_id="gpt-3.5-turbo",
    api_key=OPENAI_API_KEY,
    model_provider="openai"
)
model_image = LiteLLMModel(
    model_id="gpt-3.5-turbo",  # Changed from DALLE to GPT for consistency
    api_key=OPENAI_API_KEY,
    model_provider="openai"
)

# Global dataset variable
DATASET = None

# Paths
DATASET_PATH = "dataset/raw_materials_inventory.csv"
DATABASE_PATH = "dataset/database.db"

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
    Answers queries about the loaded dataset using GPT.

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
        """
        
        prompt = f"""Based on the following dataset:
        {dataset_info}
        
        Context: {context if context else 'None'}
        
        Please answer: {query}
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
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
        if "plot" in query.lower() or "chart" in query.lower():
            column_name = query.split()[-1]
            if column_name in DATASET.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=DATASET, x=column_name, kde=True, ax=ax)
                plt.title(f'Distribution of {column_name}')
                plt.xlabel(column_name)
                plt.ylabel('Frequency')
                return fig
            else:
                return None
        else:
            return None
    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")
        return None

# Define Agents
agent_csv = CodeAgent(tools=[load_dataset_from_csv], model=model_query)
agent_db = CodeAgent(tools=[load_dataset_from_db], model=model_query)
agent_query = CodeAgent(tools=[query_dataset], model=model_query)
agent_plot = CodeAgent(tools=[generate_plot], model=model_image)

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