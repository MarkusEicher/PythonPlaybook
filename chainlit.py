import chainlit as cl
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

model_id = "tiiuae/falcon-7b-instruct"
falcon_llm = HuggingFaceHub(
    repo_id = model_id,
    model_kwargs = {"temperature":0.3, "max_new_tokens":2000},
    huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_KEY")
)

template = """
    You are an AI Coaching Training Assistant in the making.

    {question}

"""
# @cl.langchain_factory(use_async=False)
# def factory():
#     prompt = PromptTemplate(template = template, input_variables = ['question'])
#     falcon_chain = LLMChain(prompt = prompt, llm = falcon_llm, verbose = True)
#     return falcon_chain

@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)

    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message:str):
    llm_chain = cl.user_session.get("llm_chain")

    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(content=res["text"]).send()
