"""Pipeline orchestrator for sequential agent execution."""

import logging
from typing import Optional
from agents.base import BaseAgent, AgentInput, AgentOutput
from config.loader import load_pipeline_config

logger = logging.getLogger(__name__)


class Pipeline:
    """Sequential pipeline that executes agents one after another.
    
    Each agent receives input and context from previous agents,
    passing its results to the next agent in the chain.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the pipeline.
        
        Args:
            config_path: Optional path to pipeline configuration file
        """
        self.config = load_pipeline_config(config_path)
        self.agents: list[BaseAgent] = []
        self.execution_history: list[AgentOutput] = []
    
    def add_agent(self, agent: BaseAgent) -> "Pipeline":
        """Add an agent to the pipeline.
        
        Args:
            agent: The agent to add
            
        Returns:
            Self for chaining
        """
        self.agents.append(agent)
        logger.info(f"Added agent: {agent.name}")
        return self
    
    def execute(self, initial_query: str, initial_context: Optional[dict] = None) -> AgentOutput:
        """Execute the pipeline with an initial query.
        
        Args:
            initial_query: The research query to start with
            initial_context: Optional initial context (can include 'parameters' dict)
            
        Returns:
            Final AgentOutput from the last agent in the pipeline
        """
        if not self.agents:
            raise ValueError("Pipeline has no agents. Add agents before executing.")
        
        context = initial_context.copy() if initial_context else {}
        current_query = initial_query
        
        # Extract parameters from context if present
        initial_params = context.pop("parameters", {})
        
        logger.info(f"Starting pipeline with query: {initial_query}")
        
        for i, agent in enumerate(self.agents):
            logger.info(f"Executing agent {i+1}/{len(self.agents)}: {agent.name}")
            
            # Merge initial parameters with agent-specific config
            agent_config_params = self.config.get("agents", {}).get(agent.name, {}).get("parameters", {})
            params = {**initial_params, **agent_config_params}
            
            input_data = AgentInput(
                query=current_query,
                context=context,
                parameters=params
            )
            
            try:
                agent.validate_input(input_data)
                output = agent.execute(input_data)
                
                # Extract events from findings and pass to next agent
                findings = output.findings
                events = findings.get("events", [])
                
                # Pass entire findings to context for next agent
                context["events"] = events
                context[agent.name] = findings
                
                # Also update summary for next agent
                if events:
                    current_query = f"Found {len(events)} events to process"
                else:
                    current_query = str(findings)
                
                self.execution_history.append(output)
                logger.info(f"Agent {agent.name} completed successfully")
                
            except Exception as e:
                logger.error(f"Agent {agent.name} failed: {e}")
                raise
        
        final_output = self.execution_history[-1]
        logger.info(f"Pipeline completed. Final output from: {final_output.agent_name}")
        
        return final_output
    
    def get_history(self) -> list[AgentOutput]:
        """Get the execution history of all agents.
        
        Returns:
            List of outputs from each agent in order
        """
        return self.execution_history
    
    def clear(self) -> None:
        """Clear the execution history."""
        self.execution_history.clear()
        logger.info("Pipeline history cleared")
