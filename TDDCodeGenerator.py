import os
from langchain.document_loaders import UnstructuredPDFLoader,PyPDFLoader
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import google.generativeai as palm
from genai.credentials import Credentials
from genai.schemas import GenerateParams
from genai.extensions.langchain import LangChainInterface
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from genai import PromptPattern

load_dotenv()
palm.configure(api_key=os.environ['API_KEY'])
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

api_key = os.getenv("GENAI_KEY", None)
api_url = os.getenv("GENAI_API", None)
creds = Credentials(api_key, api_endpoint=api_url) # credentials object to access the LLM service
params = GenerateParams(decoding_method="greedy", max_new_tokens=1000)

loader = PyPDFLoader("C://Users//00028Z744//Downloads//generative-solutions-tdd//test_tdd.pdf")
pdfData = loader.load()
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
splitData = text_splitter.split_documents(pdfData)

defaults = {
  'model': 'models/text-bison-001',
  'temperature': 0.1,
  'candidate_count': 1,
  'top_k': 40,
  'top_p': 0.95,
  'max_output_tokens': 1024,
}

input_text = ''.join(splitData[0].page_content)

prompt = """Transform a sentence into a bulleted list.
Sentence: I got up, went to the beach, surfed for a bit, and then took a long drive back with the windows down.
Bulleted: • I got up
• Went to the beach
• Surfed for a bit
• Took a long drive back with the windows down
Sentence: """ +input_text+ """
Bulleted:"""


response = palm.generate_text(
  **defaults,
  prompt=prompt
)
print(response.result)

template = PromptPattern.from_str(response.result+"{{language}}")
langchain_model = LangChainInterface(model="bigcode/starcoder", params=params, credentials=creds)
first_prompt = template.langchain.as_template()

first_chain = LLMChain(llm=langchain_model, prompt=first_prompt) #model + prompt template


overall_chain = SimpleSequentialChain(chains=[first_chain], verbose=True)
generated_code = overall_chain.run("java")
print(generated_code)

#checkpoint = "bigcode/starcoder"
#device = "cpu"
#tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#model = AutoModelForCausalLM.from_pretrained(checkpoint,low_cpu_mem_usage=True)

#inputs = tokenizer.encode(response, return_tensors="pt")
#outputs = model.generate(inputs)
#print(tokenizer.decode(outputs[0]))
