"""
MemGPT - Memory-Augmented Language Model Agent with Advanced RAG
Main entry point and example usage
"""

import logging
from utils.logging_config import setup_logging
from agent.rag_graph import MemGPTRAGAgent
from database.connection import db

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

def main(agent_id = "user_demo_agent"):
    """Main function to run MemGPT RAG agent"""
    
    print("=" * 60)
    print("MemGPT with Advanced Agentic RAG")
    print("=" * 60)
    print()
    
    # Test database connection
    print("Testing database connection...")
    if db.test_connection():
        print("✓ Database connected successfully\n")
    else:
        print("✗ Database connection failed!")
        print("Please check your connection string in config.py\n")
        return
    
    # Initialize agent
    agent_id = agent_id
    print(f"Initializing MemGPT RAG agent: {agent_id}")
    
    try:
        agent = MemGPTRAGAgent(agent_id=agent_id)
        print("✓ Agent initialized with RAG capabilities\n")
        print("Available tools: Memory management + Document upload + Web search")
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        logger.error(f"Agent initialization error: {e}", exc_info=True)
        return
    
    # Interactive chat loop
    print("\nChat with MemGPT (type 'quit' to exit, 'memory' to view core memory)")
    print("-" * 60)
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == 'memory':
                # Display core memory
                core_memory = agent.memory_manager.get_core_memory()
                print("\n" + "=" * 60)
                print("CORE MEMORY")
                print("=" * 60)
                print(f"\nHuman Persona:\n{core_memory['human_persona']}")
                print(f"\nAgent Persona:\n{core_memory['agent_persona']}")
                print(f"\nCore Facts: {core_memory['facts']}")
                print("=" * 60 + "\n")
                continue
            
            # Get agent response
            print("\nMemGPT: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
            print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            logger.error(f"Chat error: {e}", exc_info=True)
            print()


if __name__ == "__main__":
    main()
