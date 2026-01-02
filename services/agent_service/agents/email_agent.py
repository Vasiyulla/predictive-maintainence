from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-70b-8192")

def generate_email(machine_id, sop):
    prompt = f"""
Draft a professional internal email to procurement.

Machine ID: {machine_id}

Maintenance SOP:
{sop}

Include spare parts and urgency.
"""
    return llm.invoke(prompt).content
