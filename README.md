# StriveMate(.v2: migrated from older version)
Personal-Generalized-Copilot

### current state: 
    - the following functionality works properly:
        - short term sliding-context-window
        - custom function call for long term relevant chat history
        - continuous efficient summarization per 550 words
        - continuos vector-database update and sqlite update summary + complete chat records.

# TASKS:
- [X] synthetic chat data generation
- [X] chat history summarizer pipeline
- [X] creating context window logic/pipeline
- [X] designing short term memory sequence and storage with Sqlite3
- [ ] Retrieval Augmented Generation:
  - [X] Vector Database implementation
  - [X] Generic Retriever Logic
  - [ ] Generate embeddings and vector store for chat history summary, include:
    - ID (time) (which will link to full-chat-history)
    - Metadata (consider what specific metadata is needed)
  - [ ] Implement RAG functionality into the `mock-chat.py` (also rename this file)

- [] finetuning t5-flan-base or large on synthetic data, using QLora-4bit and Peft
- [] push the model to huggingface hub.
- [] creating FastAPI endpoints
- [] deploying model and vector databases to cloud (sagemaker, EC2, lambda)
- [] embedding finetuned model instead of gpt api call
- [] testing functionality
- [] adding key-information memory with vector database
- [] further developments
- [] adding CI/CD pipeline with github actions
- [] adding unit testing
- Research Agent:
  - [] add youtube video searcher with youtube data api
  - [] add google searcher, wikipedia searcher
  - [] we may better to use the finetuned model for this, after SPR implementation, or 
       since this is a complex task, just call gpt3.5

## RESOURCE:
- https://huggingface.co/google/flan-t5-small
- https://github.com/google-research/text-to-text-transfer-transformer
- https://colab.research.google.com/drive/1nxghpO7UzB0VgiVAH-_s-4m7g2_qYEgd?usp=sharing
- https://huggingface.co/datasets/samsum
- https://youtu.be/PZE_08Lshr4?si=Eb8XwgScZGCV-wqd
- https://youtu.be/ypzmPwLH_Q4?si=XKKbM3VCIoU46TbU

# CAUTION:
- Do not put quotes inside prompt yaml -> dict. And after loading the yaml -> json.dump()
- limit the summary length to 50, since the context window is short(400-500) already.


### CREDITS:
- the SPR(sparse priming representation) prompting idea is inspired by David Shapiro (https://github.com/daveshap)
