import os
import json
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from pydantic import BaseModel, Field, ValidationError

from dotenv import load_dotenv

load_dotenv()

# Keys
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


class State(TypedDict):
    messages: Annotated[
        list[BaseMessage], add_messages
    ]  # Be explicit about list[BaseMessage]


evidence_string = """
    Evidence of one of the following (each field should be unique): A receipt of lesser nationally or internationally recognized prizes or awards for excellence in the field of endeavor, 
        membership in associations in the field, which require outstanding achievements as judged by recognized national or international experts;
        published material about the alien in professional or major trade publications or other major media,
        participation on a panel or individually as a judge of others’ work in the field or a related field,
        original scientific, scholarly, artistic, athletic, or business-related contributions of major significance in the field,
        authorship of scholarly articles in the field in professional or major trade publications or other major media,
        display of the alien’s work at artistic exhibitions or showcases,
        evidence that the alien has performed in a leading or critical role for organizations or establishments that have distinguished reputations,
        evidence that the alien has commanded a high salary or other high compensation for services,
        evidence of commercial successes in the performing arts as shown by box office receipts or music or video sales.
"""


class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""

    achievement_evidence: str = Field(
        description="Evidence of a one-time achievement (for example, a major internationally recognized award)"
    )
    achievement_evidence_url: str = Field(
        description="URL for the 'achievement_evidence'"
    )
    field_one: str = Field(description=evidence_string)
    field_one_url: str = Field(description="URL for the 'field_one'")
    field_two: str = Field(description=evidence_string)
    field_two_url: str = Field(description="URL for the 'field_two'")
    field_three: str = Field(description=evidence_string)
    field_three_url: str = Field(description="URL for the 'field_three'")


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
            You are an immigration paralegal that does not give legal advice. 
            You are looking for the following information on the subject provided to you by the user:

            1. Evidence of a one-time achievement (for example, a major internationally recognized award); or
            2. At least three of the following:
                a. Receipt of lesser nationally or internationally recognized prizes or awards for excellence in the field of endeavor;
                b. Membership in associations in the field, which require outstanding achievements as judged by recognized national or international experts;
                c. Published material about the alien in professional or major trade publications or other major media;
                d. Participation on a panel or individually as a judge of others’ work in the field or a related field;
                e. Original scientific, scholarly, artistic, athletic, or business-related contributions of major significance in the field;
                f. Authorship of scholarly articles in the field in professional or major trade publications or other major media;
                g. Display of the alien’s work at artistic exhibitions or showcases;
                h. Evidence that the alien has performed in a leading or critical role for organizations or establishments that have distinguished reputations;
                i. Evidence that the alien has commanded a high salary or other high compensation for services; and
                j. Evidence of commercial successes in the performing arts as shown by box office receipts or music or video sales.
        """
        ),
        ("placeholder", "{messages}"),
    ]
)


graph_builder = StateGraph(State)

with open("/code/src/fields.json", "r") as f:
    site_data = json.load(f)
    sites = [field["domains"] for field in site_data["journals"]]
    sites = [item for sublist in sites for item in sublist]
    sites.append(site_data["additional_general_academic_resources"])
    sites.append(site_data["general_news_sources"])

tavily = TavilySearch(api_key=TAVILY_API_KEY, included_domains=sites)
tools = [tavily]

llm = ChatOpenAI(
    model_name="deepseek/deepseek-chat-v3-0324:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OPENROUTER_API_KEY,
)

llm_with_tools = llm.bind_tools(tools).with_structured_output(ResponseFormatter)


async def chatbot(state: State):
    # Invoke the LLM to get the structured output
    formatted_messages = prompt.format_messages(messages=state["messages"])
    structured_output = await llm_with_tools.ainvoke(formatted_messages)

    # Decide how to represent this in the chat history
    # You could create an AIMessage from it, or just store it separately.
    # For simplicity, let's create an AIMessage with a string representation
    ai_message = AIMessage(content=f"{structured_output.model_dump_json(indent=2)}")

    return {
        "messages": [ai_message],  # Add the AI message to history
    }


async def stream_graph_updates(user_input: str):
    async for event in graph.astream(
        {"messages": [HumanMessage(content=user_input)]}
    ):  # Use HumanMessage directly
        for value in event.values():
            if "messages" in value and value["messages"]:
                return json.loads(value["messages"][-1].content)

    return None


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


async def run_graph(user_input: str):
    while True:
        try:
            return await stream_graph_updates(user_input)
        except ValidationError as e:
            continue
        except Exception as e:
            print(e)
            break
