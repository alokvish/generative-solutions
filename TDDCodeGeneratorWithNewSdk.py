import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from jira import JIRA
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("JIRA_KEY",None)
url = os.getenv("JIRA_URL",None)
user = os.getenv("JIRA_USER",None)
watsonx_key=os.getenv("IBM_WATSONX_API_KEY",None)
watsonx_url=os.getenv("IBM_WATSONX_URL",None)
watsonx_project_id=os.getenv("IBM_WATSONX_PROJECT_ID",None)

jira = JIRA(options={'server': url}, basic_auth=(user, key))
singleIssue = jira.issue('GEN-1')
print('{}: {}:{}'.format(singleIssue.key,
                         singleIssue.fields.summary,
                         singleIssue.fields.description,
                         singleIssue.fields.reporter.displayName))

star_coder_model = Model(
    model_id=ModelTypes.STARCODER,
    credentials={
        "apikey": watsonx_key,
        "url": watsonx_url
    },
    project_id=watsonx_project_id
    )

#prompt_template = "As a user I want a function that adds two numbers and returns the result using {language}?"
prompt_template = singleIssue.fields.description + "using"+ "{language}"
llm_chain = LLMChain(llm=star_coder_model.to_langchain(), prompt=PromptTemplate.from_template(prompt_template))
chain_response = llm_chain.run("python")
print(chain_response)