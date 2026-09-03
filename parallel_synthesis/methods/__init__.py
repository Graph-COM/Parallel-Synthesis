from typing import List


class Agent:
    def __init__(self, name: str, role: str) -> None:
        self.name = name
        self.role = role


def default_agents(num_parallel_agents: int = 3) -> List[Agent]:
    count = max(1, int(num_parallel_agents))
    agents = [
        Agent(name=f"Sub-agent {idx}", role=f"sub_agent_{idx}")
        for idx in range(1, count + 1)
    ]
    agents.append(Agent(name="Judger", role="judger"))
    return agents


__all__ = ["Agent", "default_agents"]
