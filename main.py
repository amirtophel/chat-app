import os
from dotenv import load_dotenv
from query import load_documents, setup_conversational_chain

load_dotenv()

def main():
    yellow, green, white = "\033[0;33m", "\033[0;32m", "\033[0;39m"
    data_directory = "data"
    
    print(f"{yellow}Loading documents...")
    documents = load_documents(data_directory)
    print("Setting up conversational chain...")
    pdf_qa = setup_conversational_chain(documents)

    chat_history = []
    print(f"{yellow}---------------------------------------------------------------------------------")
    print('Welcome to the ChatBot. Type your query below. To exit, type "exit", "quit", "q", or "f".')
    print('---------------------------------------------------------------------------------')
    
    while True:
        query = input(f"{green}Prompt: ")
        if query.lower() in ["exit", "quit", "q", "f"]:
            print('Exiting...')
            break
        if query == '':
            continue
        
        try:
            result = pdf_qa.invoke({"question": query, "chat_history": chat_history})
            print(f"{white}Answer: " + result["answer"])
            chat_history.append((query, result["answer"]))
            if len(chat_history) > 5:
                chat_history = chat_history[-5:]
        except Exception as e:
            print(f"An error occurred while processing your query: {e}")

if __name__ == "__main__":
    main()
