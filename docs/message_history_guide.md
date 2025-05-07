# Using PydanticAI Message History for Agent Communication

This guide explains how to use PydanticAI's message history features to enable continuous and coherent conversations between agents in your iterative research projects.

## Overview

PydanticAI provides a powerful way to maintain conversation history across multiple agent runs. This allows agents to:

1. **Remember past interactions** - Agents can refer to information from previous conversations
2. **Build on prior knowledge** - Each iteration can build on insights from previous iterations
3. **Maintain conversational context** - The system can provide more relevant and contextual responses
4. **Persist research sessions** - You can save and resume research projects across sessions

## How It Works

The implementation in our `IterativeAgent` uses these key features:

1. **Message History Storage**: Each agent stores its conversation history
2. **Contextual Messaging**: Messages are passed between iterations to maintain context
3. **Serialization**: Message histories can be saved to disk and loaded later
4. **Stateful Research**: The entire research state (including all agent conversations) can be persisted

## Using Message History in Agents

### Key Components

- `message_history`: Parameter passed to agent runs to maintain context
- `all_messages()`: Method to retrieve all messages exchanged during an agent run
- `new_messages()`: Method to get just the new messages from the last run
- `ModelMessagesTypeAdapter`: Type adapter used for serializing/deserializing messages

### Example: Basic Agent Communication

```python
from pydantic_ai import Agent

# Initialize an agent
agent = Agent('openai:gpt-4o', system_prompt='You are a research assistant.')

# First run - initial query
result1 = agent.run_sync('What are the environmental impacts of EVs?')

# Second run - follow-up query that references the first
result2 = agent.run_sync('Compare that with hybrid vehicles.', message_history=result1.new_messages())

# Get all messages exchanged
all_messages = result2.all_messages()
```

## Implementation in IterativeAgent

In `IterativeAgent`, we've implemented message history as follows:

1. The `ResearchState` class now has an `agent_messages` dictionary that stores message histories for each agent
2. When running agents, we pass previous message history if available
3. After each agent run, we store the updated message history back in the state
4. We've added methods to serialize and deserialize the entire research state

### Key Changes

```python
# Store message history for each agent
state.agent_messages = {
    "thinking_agent": [],
    "knowledge_gap_agent": [],
    # ... other agents
}

# Pass message history when running an agent
result = await agent.run(
    input_prompt,
    message_history=state.agent_messages["agent_name"] if state.agent_messages["agent_name"] else None
)

# Store updated message history
state.agent_messages["agent_name"] = result.all_messages()
```

## Persisting Research Sessions

One of the most powerful aspects of using message history is the ability to save and resume research sessions:

```python
# Save a research session
agent.export_session(agent.research_state, filepath="research_session.json")

# Load a research session
loaded_agent, loaded_state = IterativeAgent.load_session_from_file("research_session.json")

# Continue research from where you left off
continued_report = await loaded_agent.continue_research(loaded_state, max_additional_iterations=3)
```

## Benefits

1. **Improved Coherence**: Agents build on previous knowledge without repetition
2. **Efficient Research**: No need to restate context in every query
3. **Long-running Research**: Research can span multiple sessions over time
4. **Better Context Understanding**: Agents understand the full context of the research

## Limitations and Considerations

1. **Token Limits**: Message histories grow with each iteration, and may hit model token limits
2. **Memory Usage**: Storing full message histories for many agents can increase memory usage
3. **Model Compatibility**: Different models have different formats, though PydanticAI handles most of this

## Advanced Usage

### Custom Message Serialization

```python
from pydantic_core import to_jsonable_python
from pydantic_ai.messages import ModelMessagesTypeAdapter

# Convert messages to serializable format
as_python_objects = to_jsonable_python(messages)

# Save to disk
import json
with open("messages.json", "w") as f:
    json.dump(as_python_objects, f)

# Load from disk
with open("messages.json", "r") as f:
    loaded_data = json.load(f)
    messages = ModelMessagesTypeAdapter.validate_python(loaded_data)
```

## Conclusion

Using PydanticAI's message history for agent communication significantly enhances the capability of your research agents. It allows for more coherent, contextual, and persistent research projects that can span multiple iterations and sessions.

For examples of the implementation, see:
- `src/iterative_research.py` - Main implementation of message history
- `examples/research_with_continuity.py` - Example of using message history with saved sessions 