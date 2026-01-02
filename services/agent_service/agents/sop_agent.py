from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-70b-8192")

def generate_sop(failure_analysis, manual_context):
    prompt = f"""
Create a clear step-by-step maintenance SOP.

Failure Analysis:
{failure_analysis}

Manual Reference:
{manual_context}
"""
    return llm.invoke(prompt).content
