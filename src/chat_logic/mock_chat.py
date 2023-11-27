import os 
import time
import json
from src.model.openai import seq_gpt_call
from src.generic import load_yaml

import openai
from src.memory.database.sqlite3 import Sqlite3Connection

from src.custom_tools import call_vector_search

from src.memory.vectorizer import VectorRetriever


#1. we should handle summarization in background, asynchronously.
#2. for the first 3 cycles, we need to ensure the previous summaries as 1 for 2nd cycle, and 2 for 3rd cycle and 3 for cycles > 3
#3. we should implement retry mechanism for the summarization and other api calls.



path_1= os.path.join(os.getcwd(), "instructions", "efficient_summary.yaml") 
path_2 = os.path.join(os.getcwd(), "instructions", "custom_function.yaml")
summarizer_prompt = load_yaml(path_1, "chat_history_summarizer")
summarizer_prompt = json.dumps(summarizer_prompt)
retrieval_prompt = json.dumps(load_yaml(path_2,"custom_function_call_prompt"))


window_token_limit = 550

chat_history = []

path = os.path.join(os.getcwd(), "chat_chunks_record.txt")

conn = Sqlite3Connection("chat_records.db", "chat_records") # connect to database and initialize the table

# INITIALIZING VECTORSTORE & VECTOR RETRIEVER
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}


persist_directory = os.path.join(os.getcwd(), "artifact" ,"vector_db")
os.makedirs(persist_directory, exist_ok=True)

collection_name= "chat_record_summaries"

vector_retriever = VectorRetriever(model_name, model_kwargs, encode_kwargs)
vector_retriever.initialize_vector_store(persist_directory, [], collection_name) # we use for inference.
print(type(vector_retriever))


def get_vector_retriever():
    global vector_retriever
    return vector_retriever

def check_token_size(text): # will be turned into token counter later.

    return len(text.split(" ")) <= window_token_limit


# Ensure that record_to_json can handle the summary argument
def record_to_json(chunk_record, id, summary=None):
    #automatically escapes double quotes to ensure that the JSON format is maintained.
    record_dict = {"id": id, "record": chunk_record}
    
    if summary is not None:
        record_dict['summary'] = summary
    #print(type(summary), type(id))

    #we can add unique id and other metadata also
    return json.dumps(record_dict, indent=4)

def save_chunk_record(chunk_record, summary="None"):
    try:
        conn.insert_record(int(time.time()), str(chunk_record), "chat_records", summary)
        print("record saved successfully")
        with open(path, "a") as f:

            f.write(chunk_record + "\n")
    except Exception as e:
        print("Error saving record:", e)



model_name = "gpt-3.5-turbo"
classic_instruction = str(retrieval_prompt)


vector_retriever = get_vector_retriever()
chunk_record = "" # sliding window of records

try:

    # this should be outside the while loop
    if conn.get_latest_record("chat_records",limit=3):
        
        last_record_summaries =  conn.get_latest_record("chat_records",limit=3) # adds latest 3 summary records, 50*3 = 150 words in total
        print(last_record_summaries)
    
    else: 
        last_record_summaries = ""

    while True: 
        #print("chunk record:", chunk_record)
        user_input = input("User: ")
        if user_input.lower() == "quit":
            if chunk_record:
                id = time.time()
                record_summary, metrics = seq_gpt_call(summarizer_prompt, chunk_record, model_name=model_name, max_tokens=50)
                print("record summary: ", record_summary)

                record = record_to_json(chunk_record, id, record_summary)
                chat_history.append(record)
                save_chunk_record(record, record_summary)
                #vector_retriever = get_vector_retriever() # not a good practice to call here
                #add this record summary as document to vector store
                vector_retriever.add_new_texts(texts=[record_summary])
                print("vector store updated with new record summary")

                print(f"Final chunk record saved for {id}")
                #print("latest record: ", conn.get_latest_record("chat_records",limit=1))
                print("\n\nExiting...")
                conn.close_connection()
            break 

        #ai_response_mock = input("AI:")
        retry = 2
        for i in range(retry):
            try: 
                #print("ai input:" , f"{last_record_summaries}\n{chunk_record}\n{user_input}")
                ai_response , metrics = seq_gpt_call(classic_instruction, f"{last_record_summaries}\n{chunk_record}\n{user_input}", model_name=model_name, max_tokens=250)
                if ai_response:
                    if "call" in ai_response: # we want to be sure not to miss any function call request
                        for i in range(retry):
                            print("calling vector search")
                            retrieved_info =" | ".join(call_vector_search(user_input, vector_retriever)) # turn list of info to a single string
                            user_input_formatted = f"###retrieved_info:\n{retrieved_info}\n###user-input:\n{user_input}" # temporary formatted input 
                            ai_response , metrics = seq_gpt_call(classic_instruction, f"{last_record_summaries}\n{chunk_record}\n{user_input_formatted}", model_name=model_name, max_tokens=250)
                        
                    print("succesfull response")
                    
                else:
                    time.sleep(3)
                    print("response failed, retrying...")
            except Exception as e :
                print("some error occured: ", e)

        print(f"AI: {ai_response}")


        concat_record = f"USER:  {user_input}\n\nAI:  {ai_response}\n\n"
        chunk_record += concat_record
        print("chunk record updated")
        
        if not check_token_size(chunk_record): # if chunk record exceeds 550 words
            print("RECORD IS BEING SAVED.. BE PATIENT...")
            id = time.time()

            for i in range(retry):
                record_summary, metrics = seq_gpt_call(summarizer_prompt, chunk_record, model_name=model_name, max_tokens=50)
                if record_summary:
                    print("record summary: ", record_summary)
                    break
                else:
                    time.sleep(3)
                    print("summarization failed, retrying...")
                    
                    

            chunk_record = chunk_record.strip().split(" ")[:window_token_limit]
            chunk_record = " ".join(chunk_record)

            record = record_to_json(chunk_record, id,summary=record_summary)
            chat_history.append(f"{id}: "+ record + "\n" + "############\nSUMMARY:" + record_summary+ "\n\n<END>\n\n")
            
            vector_retriever.add_new_texts(texts=[record_summary])
            print("vector store updated with new record summary")

            record_summary = record_summary.strip().replace("\n"," ")
            save_chunk_record(record, record_summary)
            print(f"record chunk saved for {id},  chunk length: {len(record.split(' '))}")
        
            last_record_summaries += f"\n{record_summary}"
            chunk_record = "" # reset the short term window, indicating that the context window should slide forward.
            
            
except KeyboardInterrupt:
    
    print("\n\nExiting...")
    conn.close_connection()


if __name__ == "__main__":
    pass 
    #vector_retriever = get_vector_retriever()
    #print(type(vector_retriever))

