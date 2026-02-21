from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define the model
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0,
        max_new_tokens=200
    )
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name: str = Field(description='NAme of the Person')
    age: int = Field(gt=18, description='Age of the Person')
    city: str = Field(description='Name of city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person) 

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.invoke({'place':'indian'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)

chain = template | model | parser

final_result = chain.invoke({'place':'sri lankan'})

print(final_result)