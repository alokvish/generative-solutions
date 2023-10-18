import os
from langchain import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader,PyPDFLoader
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

loader = PyPDFLoader("TDD_Sample.pdf")

pdfData = loader.load()
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
splitData = text_splitter.split_documents(pdfData)

print(splitData)

checkpoint = "bigcode/starcoder"
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#model = AutoModelForCausalLM.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint,low_cpu_mem_usage=True, max_shard_size="200MB").to(device)

inputs = tokenizer.encode(splitData, return_tensors="pt").to(device)
#inputs = tokenizer.encode(splitData, return_tensors="pt")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))