from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from operator import add as add_messages
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

# ---------------------------
# LLM Setup
# ---------------------------

# Data Gathering LLM
data_llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.1,
    max_new_tokens=256,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    return_full_text=False,   # ✅ must be passed directly
    do_sample=False           # ✅ must be passed directly
)

# Writing LLM
writer_llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.3,
    max_new_tokens=512,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    return_full_text=False,   # ✅ must be passed directly
    do_sample=True            # ✅ must be passed directly
)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# PDF + VectorDB Setup
# ---------------------------

pdf_path = r"E:\projects\RAG_agent\MCIOT_LAB.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"✅ PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"❌ Error loading PDF: {e}")
    raise

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)

persist_directory = r"E:\projects\RAG_agent\chroma_db"
collection_name = "mciot_lab"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print("✅ Created ChromaDB vector store!")
except Exception as e:
    print(f"❌ Error setting up ChromaDB: {str(e)}")
    raise

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ---------------------------
# Tools
# ---------------------------

@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns information from the MCIOT_LAB document."""
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information in the MCIOT_LAB document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)

tools = [retriever_tool]

# ---------------------------
# Agent State
# ---------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    gathered_data: str
    needs_data: bool

# ---------------------------
# Agent Functions
# ---------------------------

def gather_data(state: AgentState) -> AgentState:
    """First LLM - focuses on understanding the query and gathering relevant data."""
    print("🔍 Data Gathering LLM is analyzing your question...")

    messages = list(state['messages'])
    user_question = messages[-1].content

    try:
        prompt_text = (
            f"Analyze this question about MCIOT lab: '{user_question}'. "
            "What specific topics should I search for in the document? Provide 2-3 key search terms."
        )

        response = data_llm.invoke(prompt_text)

        if not response or len(response.strip()) == 0:
            response = f"Need to search for information about: {user_question}"

        print(f"📝 Data Gatherer says: {response}")

        ai_message = AIMessage(content=response)

        new_state = {
            'messages': state['messages'] + [ai_message],
            'needs_data': True,
            'gathered_data': state.get('gathered_data', '')
        }

        return new_state
    except Exception as e:
        print(f"❌ Error in data gathering: {e}")
        fallback_message = AIMessage(content=f"Analyzing question about: {user_question}")
        return {
            'messages': state['messages'] + [fallback_message],
            'needs_data': True,
            'gathered_data': state.get('gathered_data', '')
        }

def execute_retrieval(state: AgentState) -> AgentState:
    """Execute the retrieval tool based on the data gathering analysis."""
    print("📚 Retrieving information from the document...")

    user_question = next((msg.content for msg in reversed(state['messages']) if isinstance(msg, HumanMessage)), "")
    search_query = user_question

    print(f"🔍 Searching for: {search_query}")

    try:
        retrieved_data = retriever_tool.invoke(search_query)
        print(f"📄 Retrieved {len(retrieved_data)} characters of data")

        tool_message = ToolMessage(
            tool_call_id="retrieval_call",
            name="retriever_tool",
            content=retrieved_data
        )

        return {
            'messages': state['messages'] + [tool_message],
            'gathered_data': retrieved_data,
            'needs_data': False
        }
    except Exception as e:
        print(f"❌ Error retrieving data: {e}")
        error_message = ToolMessage(
            tool_call_id="retrieval_error",
            name="retriever_tool",
            content=f"Error retrieving information: {str(e)}"
        )
        return {
            'messages': state['messages'] + [error_message],
            'gathered_data': "",
            'needs_data': False
        }

def generate_response(state: AgentState) -> AgentState:
    """Second LLM - focuses on writing a comprehensive response using gathered data."""
    print("✍️ Writing LLM is generating your response...")

    user_question = next((msg.content for msg in reversed(state['messages']) if isinstance(msg, HumanMessage)), "")
    gathered_data = state.get('gathered_data', '')

    if gathered_data and len(gathered_data.strip()) > 0:
        writing_prompt = f"""Question: {user_question}

Document Information:
{gathered_data[:2000]}

Based on the document information above, provide a detailed answer to the question. Be specific and cite relevant parts."""
    else:
        writing_prompt = f"Answer this question about MCIOT lab: {user_question}"

    try:
        response = writer_llm.invoke(writing_prompt)

        if not response or len(response.strip()) == 0:
            response = (
                f"Based on the MCIOT_LAB document, regarding your question about '{user_question}': "
                "I found relevant information but encountered an issue generating the response."
            )

        print(f"✅ Generated response of {len(response)} characters")

        ai_message = AIMessage(content=response)

        return {
            'messages': state['messages'] + [ai_message],
            'gathered_data': gathered_data,
            'needs_data': False
        }
    except Exception as e:
        print(f"❌ Error in response generation: {e}")
        if gathered_data:
            fallback_response = f"Based on the MCIOT_LAB document, here's what I found regarding '{user_question}':\n\n{gathered_data[:1000]}..."
        else:
            fallback_response = f"I apologize, but I encountered an error generating a response for your question: '{user_question}'."

        ai_message = AIMessage(content=fallback_response)
        return {
            'messages': state['messages'] + [ai_message],
            'gathered_data': gathered_data,
            'needs_data': False
        }

# ---------------------------
# Graph Setup
# ---------------------------

graph = StateGraph(AgentState)

graph.add_node("data_gatherer", gather_data)
graph.add_node("retriever", execute_retrieval)
graph.add_node("writer", generate_response)

graph.add_conditional_edges(
    "data_gatherer",
    lambda state: state.get('needs_data', True),
    {True: "retriever", False: "writer"}
)

graph.add_edge("retriever", "writer")
graph.add_edge("writer", END)

graph.set_entry_point("data_gatherer")

rag_agent = graph.compile()

# ---------------------------
# Run Loop
# ---------------------------

def running_agent():
    print("\n=== DUAL-LLM RAG AGENT ===")
    print("🤖 Data Gathering LLM: HuggingFaceH4/zephyr-7b-beta")
    print("✍️  Writing LLM: HuggingFaceH4/zephyr-7b-beta")
    print("Ask questions about the MCIOT_LAB document. Type 'exit' or 'quit' to stop.")
    print("-" * 60)

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        try:
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "gathered_data": "",
                "needs_data": True
            }

            print("\n" + "="*60)
            result = rag_agent.invoke(initial_state)

            print("\n=== FINAL ANSWER ===")
            final_message = result['messages'][-1]
            if hasattr(final_message, 'content'):
                print(final_message.content)
            else:
                print("Sorry, I couldn't generate a proper response.")
            print("="*60)

        except Exception as e:
            print(f"\n❌ Error processing your question: {e}")
            print("This might be a temporary API issue. Please try again.")

            try:
                print("\n🔄 Trying direct search as fallback...")
                docs = retriever.invoke(user_input)
                if docs:
                    print("\n=== DIRECT SEARCH RESULTS ===")
                    for i, doc in enumerate(docs[:2], 1):
                        print(f"\n📄 Document {i}:")
                        print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                else:
                    print("No relevant documents found.")
            except Exception as fallback_error:
                print(f"Fallback search also failed: {fallback_error}")

if __name__ == "__main__":
    running_agent()
