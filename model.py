from langchain_cohere import ChatCohere
def maze_ai(query):
    llm = ChatCohere(
        model="command-r-08-2024",
        temperature=0,
        cohere_api_key="YOUR _API_KEY_HERE",
        max_tokens=10,
        )
    return llm.invoke(query).content