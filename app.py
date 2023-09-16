import streamlit as st

pdfpath = '/home/lung/Documents/google data/'
savepath = 'app_index'

global chain

def createVectorStore():
    ############ Load pdf documents into memory #############

    #from langchain.document_loaders import DirectoryLoader

    from langchain.document_loaders import PyPDFDirectoryLoader
    loader = PyPDFDirectoryLoader(
    path=pdfpath, # my local directory
    glob='**/*.pdf', # we only get pdfs
    #show_progress=True
    )
    docs = loader.load()

    ######################## split PDF into chunks ################

    from langchain.text_splitter import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
    )

    #from langchain.text_splitter import RecursiveCharacterTextSplitter

    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(docs)

    ##############################Creating Embeddings and Storing in Faiss Vector Store###################3

    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # storing embeddings in the vector store
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    vectorstore.save_local(savepath) # save vector database locally

    print('Done!')
    st.experimental_rerun()

if 'chain' not in st.session_state:
    st.session_state['chain'] = None

@st.cache_resource
def loadModel():
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

    ####################stoping critiria#############

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

    ##############################Creating Embeddings and Storing in Faiss Vector Store###################3

    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # Lord the vector store from SSD

    vectorstore = FAISS.load_local("test_index", embeddings)
    ####################################Initializing Chain #######################################3

    from langchain.chains import ConversationalRetrievalChain

    st.session_state['chain'] = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)



st.title("Knowledge Search")

pdfpath = st.text_input(label="Path to PDFs")
savepath = st.text_input(label="Directory Path to Save to")

st.button("Create VectorStore", on_click=createVectorStore)

st.button("Load Model", on_click=loadModel)

llm_chat_history = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me!"):
    if st.session_state['chain'] == None:
        st.write("Model not loaded!")
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        result = st.session_state['chain']({"question": prompt, "chat_history": llm_chat_history})
        llm_chat_history = [(prompt, result["answer"])]

        response = result['answer']
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})