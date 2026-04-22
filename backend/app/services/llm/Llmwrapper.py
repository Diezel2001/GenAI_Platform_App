# from langchain_community.llms import Ollama

# llm = Ollama(
#     model="mistral",
#     base_url="http://localhost:11434"
# )


from langchain_ollama import OllamaLLM

# And initialize like this:
llm = OllamaLLM(model="mistral")

# Example interaction
# response = llm.invoke("list down 5 fill in the blank quesios where in the blank has the format '<fill>'")
