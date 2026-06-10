from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Explain the following joke - {text}", input_variables=["text"]
)

chain = RunnableSequence(prompt, model, parser, prompt1, model, parser)

result = chain.invoke({"topic": "Chess"})

print(result)
