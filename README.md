# Local Domain Knowledge Chat
### Meta LLama 2.0 and FAISS with HuggingFace and LangChain

Note: Only the `Create Vector Store` functionality works in the Streamlit app as of now. The chat session must be run in terminal.

### Requirements
Run on local Ubuntu 20.04 machine with Nvidia RTX 3090 Conda environment.

`pip install -U transformers accelerate einops langchain xformers bitsandbytes faiss-gpu sentence_transformers`

`pip3 install streamlit beautifulsoup4 newspaper3k fpdf`

Make sure to create a text file: `token.txt` in the root directory of this repository and paste in a HuggingFace access token.

### Running
To create a new Vector Database store, run `streamlit run app.py` to initiate the Streamlit app UI.
Enter valid paths to a folder of PDFs as the source and a destination folder name and click the `Create Vector Store` button.

To run a chat session, open terminal and run
`python llama2chatlocalsession.py`.
