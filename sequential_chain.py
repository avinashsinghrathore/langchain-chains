from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 3 pointer summary from the following text \n {text}",
    input_variables=["text"],
)

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro", task="text-generation")

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "China manufacturing growth"})

print(result)
