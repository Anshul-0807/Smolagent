from smolagents import CodeAgent, LiteLLMModel, tool, GradioUI
import openai
import os

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 1: Define the LLM Model (GPT-4)
model = LiteLLMModel(model_id="gpt-4", api_key=OPENAI_API_KEY)

# Step 2: Define a tool that answers user queries
@tool
def answer_query(query: str) -> str:
    """
    Processes and returns an answer for a given user query using GPT-4.

    Args:
        query: The user's question or input that needs to be processed. This can be any text-based
              query that you want GPT-4 to respond to.

    Returns:
        A string containing GPT-4's response to the user's query.

    Example:
        response = answer_query("What is machine learning?")
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Provide clear and concise answers."
                },
                {
                    "role": "user", 
                    "content": query
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Step 3: Create another tool for specific tasks
@tool
def summarize_text(text: str, max_length: int = 150) -> str:
    """
    Summarizes the given text using GPT-4.

    Args:
        text: The input text that needs to be summarized. This can be any length of text
              that requires summarization.
        max_length: The maximum number of words desired in the summary. Defaults to 150 words.

    Returns:
        A concise summary of the input text.

    Example:
        summary = summarize_text("Long article text here...", max_length=100)
    """
    try:
        prompt = f"Please summarize the following text in no more than {max_length} words:\n\n{text}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a summarization expert. Provide clear, concise summaries."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

# Step 4: Create the Agent with multiple tools
agent = CodeAgent(
    tools=[answer_query, summarize_text],
    model=model
)

# Step 5: Create and launch the Gradio UI
def main():
    # Create and launch the UI
    ui = GradioUI(agent)
    ui.launch()  # Removed custom parameters to use defaults

if __name__ == "__main__":
    main()