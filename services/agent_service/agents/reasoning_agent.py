from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-70b-8192")

def analyze_failure(sensor_summary):
    prompt = f"""
You are an expert induction motor maintenance engineer.

Given the sensor condition:
{sensor_summary}

Identify:
1. Probable failure cause
2. Affected component
3. Failure severity
"""
    response = llm.invoke(prompt)
    return response.content
