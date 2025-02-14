# # SECOND CODEBASE

# from smolagents import CodeAgent, LiteLLMModel, tool, GradioUI
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sqlite3
# from typing import Optional
# import openai  # FIX: Import OpenAI module
# import io
# import base64
# from PIL import Image

# # Load OpenAI API key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Define LLM Models
# model_query = LiteLLMModel(model_id="phi-4", api_key=OPENAI_API_KEY)
# model_image = LiteLLMModel(model_id="dalle", api_key=OPENAI_API_KEY)

# # Global dataset variable
# DATASET = None

# # Paths
# DATASET_PATH = "dataset/raw_materials_inventory.csv"
# DATABASE_PATH = "dataset/database.db"

# # Tool to load dataset from CSV
# @tool
# def load_dataset_from_csv() -> str:
#     global DATASET
#     try:
#         DATASET = pd.read_csv(DATASET_PATH)
#         return f"Dataset loaded successfully from {DATASET_PATH}. Columns: {', '.join(DATASET.columns)}"
#     except Exception as e:
#         return f"Error loading dataset from CSV: {str(e)}"

# # Tool to load dataset from a database
# @tool
# def load_dataset_from_db() -> str:
#     global DATASET
#     try:
#         conn = sqlite3.connect(DATABASE_PATH)
#         query = "SELECT * FROM raw_materials_inventory;"  # Ensure this table exists in the DB
#         DATASET = pd.read_sql_query(query, conn)
#         conn.close()
#         return f"Dataset loaded from database. Columns: {', '.join(DATASET.columns)}"
#     except Exception as e:
#         return f"Error loading dataset from database: {str(e)}"

# # Query dataset
# @tool
# def query_dataset(query: str, context: Optional[str] = None) -> str:
#     global DATASET
#     if DATASET is None:
#         return "Dataset not loaded. Please load the dataset first."
    
#     try:
#         dataset_info = f"""
#         Dataset Info:
#         - Columns: {', '.join(DATASET.columns)}
#         - Sample Data:
#         {DATASET.head(3).to_string()}
#         """
        
#         prompt = f"""Based on the following dataset:
#         {dataset_info}
        
#         Context: {context if context else 'None'}
        
#         Please answer: {query}
#         """
        
#         response = openai.ChatCompletion.create(
#             model="gpt-4",  # FIX: Use a correct OpenAI model
#             messages=[
#                 {"role": "system", "content": "You are an expert data analyst."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.3,
#             max_tokens=500
#         )
#         return response["choices"][0]["message"]["content"]
#     except Exception as e:
#         return f"Error processing query: {str(e)}"

# # Generate a plot or chart
# @tool
# def generate_plot(query: str) -> str:
#     global DATASET
#     if DATASET is None:
#         return "Dataset not loaded. Please load the dataset first."
    
#     try:
#         if "plot" in query.lower() or "chart" in query.lower():
#             column_name = query.split()[-1]
#             if column_name in DATASET.columns:
#                 plt.figure(figsize=(10, 6))
#                 sns.histplot(DATASET[column_name], kde=True)
#                 plt.title(f'Distribution of {column_name}')
#                 plt.xlabel(column_name)
#                 plt.ylabel('Frequency')
                
#                 # Save plot to a buffer
#                 img_buf = io.BytesIO()
#                 plt.savefig(img_buf, format='png')
#                 plt.close()
                
#                 img_buf.seek(0)
#                 base64_image = base64.b64encode(img_buf.getvalue()).decode('utf-8')
#                 return f"data:image/png;base64,{base64_image}"
#             else:
#                 return f"Column '{column_name}' not found in dataset."
#         else:
#             return "Query doesn't request a plot or chart."
#     except Exception as e:
#         return f"Error generating plot: {str(e)}"

# # Define Agents
# agent_csv = CodeAgent(tools=[load_dataset_from_csv], model=model_query)
# agent_db = CodeAgent(tools=[load_dataset_from_db], model=model_query)
# agent_query = CodeAgent(tools=[query_dataset], model=model_query)
# agent_plot = CodeAgent(tools=[generate_plot], model=model_image)

# # Coordinator function
# class MultiAgentCoordinator:
#     def __init__(self):
#         self.agent_csv = agent_csv
#         self.agent_db = agent_db
#         self.agent_query = agent_query
#         self.agent_plot = agent_plot

#     def handle_user_query(self, user_query: str, use_db: bool = False):
#         if use_db:
#             load_response = self.agent_db.run("load_dataset_from_db")
#         else:
#             load_response = self.agent_csv.run("load_dataset_from_csv")
#         print(load_response)
        
#         if "plot" in user_query.lower() or "chart" in user_query.lower():
#             plot_response = self.agent_plot.run("generate_plot", query=user_query)
#             return plot_response
#         else:
#             answer = self.agent_query.run("query_dataset", query=user_query)
#             return answer

# # Gradio UI
# coordinator = MultiAgentCoordinator()

# def main():
#     ui = GradioUI(coordinator)
#     ui.launch()

# if __name__ == "__main__":
#     main()


# //////////////////////////////////////////////////////////////////////////



from smolagents import CodeAgent, LiteLLMModel, tool, GradioUI
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from typing import Optional, Iterator
import openai
import io
import base64
from PIL import Image

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define LLM Models
model_query = LiteLLMModel(model_id="phi-4", api_key=OPENAI_API_KEY)
model_image = LiteLLMModel(model_id="dalle", api_key=OPENAI_API_KEY)

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
    Answers queries about the loaded dataset using GPT-4.

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
            model="gpt-4",
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
def generate_plot(query: str) -> str:
    """
    Generates a plot based on the user's query about the dataset.

    Args:
        query: The user's request for a specific plot or visualization

    Returns:
        str: A base64-encoded image string of the generated plot

    Example:
        plot = generate_plot("Create a plot for inventory levels")
    """
    global DATASET
    if DATASET is None:
        return "Dataset not loaded. Please load the dataset first."
    
    try:
        if "plot" in query.lower() or "chart" in query.lower():
            column_name = query.split()[-1]
            if column_name in DATASET.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(DATASET[column_name], kde=True)
                plt.title(f'Distribution of {column_name}')
                plt.xlabel(column_name)
                plt.ylabel('Frequency')
                
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png')
                plt.close()
                
                img_buf.seek(0)
                base64_image = base64.b64encode(img_buf.getvalue()).decode('utf-8')
                return f"data:image/png;base64,{base64_image}"
            else:
                return f"Column '{column_name}' not found in dataset."
        else:
            return "Query doesn't request a plot or chart."
    except Exception as e:
        return f"Error generating plot: {str(e)}"

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

    def run(self, task: str, stream: bool = False, reset: bool = False, additional_args: dict = None) -> Iterator[str]:
        """
        Handles user queries by coordinating between different agents.
        
        Args:
            task: The user's query or task
            stream: Whether to stream the response (required by GradioUI)
            reset: Whether to reset agent memory (required by GradioUI)
            additional_args: Additional arguments (required by GradioUI)
            
        Returns:
            Iterator[str]: Response from the appropriate agent
        """
        # Extract use_db from additional args if provided
        use_db = additional_args.get('use_db', False) if additional_args else False
        
        try:
            # Load data
            if use_db:
                load_response = self.agent_db.run("load_dataset_from_db")
            else:
                load_response = self.agent_csv.run("load_dataset_from_csv")
            print(load_response)
            
            # Handle query based on type
            if "plot" in task.lower() or "chart" in task.lower():
                result = self.agent_plot.run(
                    "generate_plot", 
                    query=task,
                    stream=stream,
                    reset=reset
                )
            else:
                result = self.agent_query.run(
                    "query_dataset", 
                    query=task,
                    stream=stream,
                    reset=reset
                )
            
            # Handle streaming vs non-streaming response
            if stream:
                for response in result:
                    yield response
            else:
                yield result
                
        except Exception as e:
            yield f"Error processing request: {str(e)}"

def main():
    coordinator = MultiAgentCoordinator()
    ui = GradioUI(coordinator)
    ui.launch()

if __name__ == "__main__":
    main()