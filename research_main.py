import json
from langgraph.graph import StateGraph, END
from AgentState import AgentState
from research_agents import (
    librarian_node, 
    historian_node, 
    critic_node, 
    archivist_node, 
    publisher_node,
    saver_node
)
import notifier

# 1. Routing Logic: The decision-making center
def router(state: AgentState):
    """
    Evaluates whether to continue, retry, or stop based on agent output.
    """
    # Check for hard failures in the error log
    if state.get("error_log"):
        print(f"⚠️ Error detected: {state['error_log'][-1]}")
    
    verdict = state.get("critic_verdict", {}).get("verdict")
    archivist_rec = state.get("archivist_report", {}).get("publish_recommendation")

    # Exit strategy: Max retries
    if state.get("iteration_count", 0) >= 3:
        print("🛑 Max iterations (3) reached. Ending session.")
        return "end"

    # Retry strategy: If Critic fails quality OR Archivist finds a duplicate
    if verdict == "fail" or archivist_rec == "reject":
        print(f"🔄 Loop Triggered: Retrying discovery... (Iteration {state['iteration_count']})")
        return "retry"
    
    return "continue"

# 2. Build the Graph
workflow = StateGraph(AgentState)

# Add all specialized nodes
workflow.add_node("librarian", librarian_node)
workflow.add_node("historian", historian_node)
workflow.add_node("critic", critic_node)
workflow.add_node("archivist", archivist_node)
workflow.add_node("publisher", publisher_node)
workflow.add_node("saver", saver_node)

# Define the Workflow Path
workflow.set_entry_point("librarian")
workflow.add_edge("librarian", "historian")
workflow.add_edge("historian", "critic")

# The Router decides if we go back to the stacks or proceed to storage
workflow.add_conditional_edges(
    "critic",
    router,
    {
        "retry": "librarian",
        "continue": "archivist",
        "end": END
    }
)

# If Archivist passes the fact, it goes through the Publisher then to the Database
workflow.add_edge("archivist", "publisher")
workflow.add_edge("publisher", "saver")
workflow.add_edge("saver", END)

# 3. Compile the System
research_app = workflow.compile()

# 4. Run the Discovery Loop
if __name__ == "__main__":
    print("✨ Starting Multi-Agent Discovery Session...")
    
    # Initialize the "clipboard" (State)
    initial_state = {
        "iteration_count": 0,
        "error_log": [],
        "search_query": "Discover a unique architectural or mythological fact.",
        "retrieved_chunks": []
    }
    
    # Run the graph
    try:
        final_output = research_app.invoke(initial_state)
        print("\n--- 🔍 Post-Mortem Analysis ---")
        #print(f"Total Iterations: {final_output.get('iteration_count')}")
        print(f"Critic Verdict: {final_output.get('critic_verdict', {}).get('verdict')}")
        print(f"Critic Issues: {final_output.get('critic_verdict', {}).get('issues')}")
        print(f"Archivist Recommendation: {final_output.get('archivist_report', {}).get('publish_recommendation')}")
        
        # Success Handling
        if final_output.get("final_notification"):
            notif = final_output["final_notification"]
            #print("\n" + "="*30)
            print("📜 FINAL DISCOVERY READY")
            #print(f"TITLE: {notif.get('ntfy_title')}")
            print(f"MESSAGE: {notif.get('ntfy_message')}")
            print(f"DATABASE ID: {notif.get('fact_id')}")
            print("="*30)
            notifier.send_to_mobile(notif.get('ntfy_message'), notif.get('fact_id'))
        else:
            print("\n🔴 Session ended without a publishable fact.")
            
    except Exception as e:
        print(f"❌ System Crash: {e}")