import uuid
from dataclasses import dataclass, field


@dataclass
class TraceContext:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str = ""
    _node_call_counts: dict = field(default_factory=dict)

    def next_execution(self, node_name: str) -> int:
        """Returns 0-based execution index for this node within the current trace."""
        n = self._node_call_counts.get(node_name, 0)
        self._node_call_counts[node_name] = n + 1
        return n

    def current_execution(self, node_name: str) -> int:
        """Read-only: returns the 0-based execution index for the node currently running.
        The handler already called next_execution() in on_chain_start, so the counter
        is already incremented — subtract 1 to get the current round index."""
        return max(0, self._node_call_counts.get(node_name, 1) - 1)

    def reset(self, thread_id: str = "") -> None:
        self.trace_id = str(uuid.uuid4())
        self.thread_id = thread_id
        self._node_call_counts = {}
