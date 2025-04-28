from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import LLM, Agent, Crew, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI()


class TopicRequest(BaseModel):
    topic: str
    temperature: float


def generate_content(topic: str, temperature: float):
    """Generates content using the LLM and search tools."""
    llm = LLM(
        model="mistral/mistral-small-latest",
        temperature=temperature,
        max_tokens=1500,
    )
    search_tool = SerperDevTool(n=10)

    senior_research_analyst = Agent(
        role="Senior Research Analyst",
        goal=f"Research and analyze the latest trends on {topic}",
        backstory="You are a senior research analyst with expertise in any topic given. You are tasked with researching and analyzing the latest trends in the industry. You will use the search tool to gather information and summarize your findings. You will also provide insights and recommendations based on your analysis.",
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
        llm=llm,
    )

    content_writer = Agent(
        role="Content Writer",
        goal="Write a comprehensive article based on the research findings while maintaining accuracy.",
        backstory="You are a content writer with expertise in topic given. You will write a comprehensive article based on research findings provided by the senior research analyst. You will ensure that the article is well-structured, informative, and engaging. You will also ensure that the article is accurate and free of errors. You will use the search tool to gather additional information if needed. You will also provide insights and recommendations based on your analysis. You will also ensure that the article is optimized.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    research_task = Task(
        description=f"""  
            1. Conduct research on the {topic} including:  
                - latest trends, challenges, and opportunities.  
                - key players and competitors.  
                - emerging technologies and innovations.  
                - market dynamics and economic factors.  
            2. Evaluate the credibility and reliability of the sources.  
            3. Summarize the findings in a clear and concise manner.  
            4. Provide insights and recommendations based on the analysis.  
            5. Include all relevant sources and references.  
        """,
        expected_output=f"""  
            A detailed research report on the latest trends in the industry, including:  
                - latest trends, challenges, and opportunities.  
                - key players and competitors.  
                - emerging technologies and innovations.  
                - market dynamics and economic factors.  
                - insights and recommendations based on the analysis.  
                - all relevant sources and references.  
                - a summary of the findings in a clear and concise manner.  
        """,
        agent=senior_research_analyst,
    )

    writing_task = Task(
        description=f"""  
            1. Write a comprehensive article based on the research findings provided by the senior research analyst.  
            2. Ensure that the article is well-structured, informative, and engaging.  
            3. Ensure that the article is accurate and free of errors.  
            4. Use the search tool to gather additional information if needed.  
            5. Provide insights and recommendations based on your analysis.  
        """,
        expected_output=f"""  
            A comprehensive article based on the research findings provided by senior research analyst, including:  
                - well-structured, informative, and engaging content.  
                - accurate and free of errors.  
                - insights and recommendations based on the analysis.  
        """,
        agent=content_writer,
    )

    crew = Crew(
        name="Industry Research Crew",
        agents=[senior_research_analyst, content_writer],
        tasks=[research_task, writing_task],
        verbose=True,
    )

    return crew.kickoff(inputs={"topic": topic})


@app.post("/generate_report/")
async def generate_report(request: TopicRequest):
    try:
        result = generate_content(request.topic, request.temperature)
        return {"report": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
