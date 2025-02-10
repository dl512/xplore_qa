import streamlit as st
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Load the dataset
sheet_id = "1G_8RMWjf0T9sNdMxKYy_Fc051I6zhdLLy6ehLak4CX4"
df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")
available = df[df["Available"] == "Y"]

# Function to create the model
def create_model(model_id):
    parameters = {
        "max_new_tokens": 256,
        "temperature": 0.5,
    }
    credentials = {
        "apikey": os.getenv("APIKEY"),
        "url": os.getenv("URL")
    }
    project_id = os.getenv("PROJECT_ID")

    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )

    return WatsonxLLM(model=model)

# Create the language model
llama_llm = create_model('meta-llama/llama-3-3-70b-instruct')
agent = create_pandas_dataframe_agent(llama_llm,
                                       available,
                                       verbose=True,
                                       allow_dangerous_code=True,
                                       handle_parsing_errors=True)

# Streamlit interface
st.title("Activity Recommendation AI")

# User input
user_question = st.text_input("Ask me about activities you might enjoy!")

# Submit button
if st.button("Submit"):
    if user_question:
        # Invoke the agent with the user's question
        msg = agent.invoke(
            [
                SystemMessage(
                    content=
                    '''
                    You are a helpful and super funny AI bot that assists a user in choosing the most suitable activities. 
                    Please respond in a concise manner with natural language. 
                    Please use [DD/MM] date format in your search. 
                    In your search, try different similar phases in both Chinese and English. 
                    Please do not use external knowledge in your answer.
                    '''
                ),
                HumanMessage(content=user_question)
            ]
        )

        output = msg['output']

        # Create a humorous response
        prompt = PromptTemplate(
            template=
            '''
            You are a very hilarious AI. 
            Based on the user's question {question}, and AI-generated response {output}, 
            please write a funny yet helpful answer. 
            Please be concise. Do not repeat the question.
            Use the same language as the user's question. 
            In your response, you should include the details of the events, including the event name, date, venue, if applicable.
            ''',
            input_variables=["question", "output"],
        )

        chain = prompt | llama_llm
        response = chain.invoke({"question": user_question, "output": output})

        # Display the response
        st.write("AI Response:")
        st.write(response)
    else:
        st.write("Please enter a question before submitting.")