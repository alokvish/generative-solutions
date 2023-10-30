import os
from langchain import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader,PyPDFLoader
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import google.generativeai as palm

load_dotenv()
palm.configure(api_key=os.environ['API_KEY'])
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

loader = PyPDFLoader("test_tdd.pdf")
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

checkpoint = "bigcode/starcoder"
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint,low_cpu_mem_usage=True)

inputs = tokenizer.encode(response, return_tensors="pt")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
