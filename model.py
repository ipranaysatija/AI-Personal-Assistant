from langchain_cohere import ChatCohere
def maze_ai(query):
    llm = ChatCohere(
        model="command-r-08-2024",
        temperature=0,
        cohere_api_key="S7NonwWcXFTLqxiHwqbtpT7HcZchJ0MA87W90XYS",
        max_tokens=10,
        )
    return llm.invoke(query).content