from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
api_key = os.getenv("API_KEY")



def get_decision_chain():
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=512
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a decision-making assistant. Based only on the provided context (policy or contract), determine if the user's case is eligible.
        If the answer is not found in the context, respond: "The answer is not present in the provided context."
        Return output in this JSON format only:
        {{{{
            "decision": "approved/rejected/unknown",
            "amount": "if mentioned then specify amount, else say 'NULL'",
            "justification": "...",
            "matched_clauses": ["clause snippet 1", "clause snippet 2"]
        }}}}
        Do not include any explanation or extra text outside the JSON.
        """),
        ("user", "Query: {question}\nContext: {context}")
    ])

    return prompt | llm



