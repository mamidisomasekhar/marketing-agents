"""Load and manage agent system prompts from configuration."""

import yaml
from pathlib import Path
from typing import Dict, Optional


class PromptManager:
    """Manages system prompts for all agents."""
    
    _instance = None
    _prompts = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._prompts is None:
            self._load_prompts()
    
    def _load_prompts(self):
        """Load prompts from YAML configuration."""
        config_path = Path(__file__).parent.parent / "config" / "agent_prompts.yaml"
        
        if not config_path.exists():
            self._prompts = {}
            return
        
        with open(config_path) as f:
            data = yaml.safe_load(f)
            self._prompts = data.get("agents", {})
    
    def get_system_prompt(self, agent_name: str) -> str:
        """Get system prompt for an agent."""
        agent_config = self._prompts.get(agent_name, {})
        return agent_config.get("system_prompt", "You are a helpful assistant.")
    
    def get_web_search_guidelines(self) -> str:
        """Get web search grounding guidelines."""
        config_path = Path(__file__).parent.parent / "config" / "agent_prompts.yaml"
        
        with open(config_path) as f:
            data = yaml.safe_load(f)
            return data.get("web_search_guidelines", "")
    
    def needs_web_search(self, agent_name: str) -> bool:
        """Check if agent requires web search grounding."""
        prompt = self.get_system_prompt(agent_name)
        return "LATEST NEWS GROUNDING" in prompt


_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """Get or create prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_system_prompt(agent_name: str) -> str:
    """Get system prompt for an agent."""
    return get_prompt_manager().get_system_prompt(agent_name)


def needs_web_search_grounding(agent_name: str) -> bool:
    """Check if agent needs web search for latest information."""
    return get_prompt_manager().needs_web_search(agent_name)
