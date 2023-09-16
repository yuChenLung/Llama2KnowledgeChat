#!pip install -qU transformers accelerate einops langchain xformers bitsandbytes faiss-gpu sentence_transformers

from torch import cuda, bfloat16
import transformers

model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
# Create a text file called token.txt and paste HF token in
file = open("token.txt", 'r')
hf_auth = str(file.read())
file.close()
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")

####################Tokenizing##############

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

####################stopping criteria#############

stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]


import torch

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]


from transformers import StoppingCriteria, StoppingCriteriaList

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

################# HF pipeline ##############################

generate_text = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

#####################HF pipeline in Langchain#####################

from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)





##############################Creating Embeddings and Storing in FAISS Vector Store###################3

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Lord the vector store from SSD

vs = ""
while vs == "":
    vs = input("Which vector store would you like to load? ")
    try:
        vectorstore = FAISS.load_local(vs, embeddings)
    except:
        print("Invalid vector store directory name.")
        vs = ""

####################################Initializing Chain #######################################3

from langchain.chains import ConversationalRetrievalChain

chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

################################## Q & A with short term memory ####################################

chat_history = [] 
query = ""
while query != 'q' and query != "quit":
    if query != "":
        chat_history = [(query, result["answer"])]

    query = input("Input: ")
    if query == 'q' or query == "quit":
        break
    result = chain({"question": query, "chat_history": chat_history})

    print(result['answer'])
    #print(result['source_documents'])


####################### see the source of the information #########################################3

#print(result['source_documents'])


