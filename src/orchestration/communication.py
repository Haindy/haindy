"""
Inter-agent communication and message passing system.
"""

import asyncio
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID

from src.core.types import AgentMessage
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class MessageType(str, Enum):
    """Types of messages that can be sent between agents."""
    
    # Control messages
    START_TEST = "start_test"
    STOP_TEST = "stop_test"
    PAUSE_TEST = "pause_test"
    RESUME_TEST = "resume_test"
    
    # Task messages
    EXECUTE_STEP = "execute_step"
    DETERMINE_ACTION = "determine_action"
    PLAN_TEST = "plan_test"
    
    # Result messages
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    ACTION_DETERMINED = "action_determined"
    PLAN_CREATED = "plan_created"
    
    # Status messages
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class MessagePriority(str, Enum):
    """Priority levels for message handling."""
    
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class MessageBus:
    """
    Central message bus for inter-agent communication.
    
    Provides publish-subscribe messaging pattern for agents to communicate
    asynchronously while maintaining loose coupling.
    """
    
    def __init__(self):
        """Initialize the message bus."""
        # Subscribers mapped by message type
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Message history for debugging and analysis
        self._message_history: List[AgentMessage] = []
        self._history_limit = 1000
        
        # Active agents registry
        self._registered_agents: Set[str] = set()
        
        # Message queues for async processing
        self._message_queues: Dict[str, asyncio.Queue] = {}
        
        # Statistics
        self._message_count: Dict[str, int] = defaultdict(int)
        
        logger.info("Message bus initialized")
    
    def register_agent(self, agent_name: str) -> None:
        """
        Register an agent with the message bus.
        
        Args:
            agent_name: Name of the agent to register
        """
        if agent_name in self._registered_agents:
            logger.warning(f"Agent {agent_name} already registered")
            return
        
        self._registered_agents.add(agent_name)
        self._message_queues[agent_name] = asyncio.Queue()
        
        logger.info(f"Agent registered: {agent_name}")
    
    def unregister_agent(self, agent_name: str) -> None:
        """
        Unregister an agent from the message bus.
        
        Args:
            agent_name: Name of the agent to unregister
        """
        if agent_name not in self._registered_agents:
            logger.warning(f"Agent {agent_name} not registered")
            return
        
        self._registered_agents.remove(agent_name)
        if agent_name in self._message_queues:
            del self._message_queues[agent_name]
        
        # Remove all subscriptions for this agent
        for message_type in self._subscribers:
            self._subscribers[message_type] = [
                sub for sub in self._subscribers[message_type]
                if not (hasattr(sub, '__self__') and 
                       hasattr(sub.__self__, 'name') and 
                       sub.__self__.name == agent_name)
            ]
        
        logger.info(f"Agent unregistered: {agent_name}")
    
    def subscribe(
        self, 
        message_type: str, 
        handler: Callable[[AgentMessage], None],
        agent_name: Optional[str] = None
    ) -> None:
        """
        Subscribe to messages of a specific type.
        
        Args:
            message_type: Type of message to subscribe to
            handler: Callback function to handle messages
            agent_name: Optional agent name for tracking
        """
        self._subscribers[message_type].append(handler)
        
        log_msg = f"Subscription added for {message_type}"
        if agent_name:
            log_msg += f" by {agent_name}"
        logger.debug(log_msg)
    
    def unsubscribe(
        self, 
        message_type: str, 
        handler: Callable[[AgentMessage], None]
    ) -> None:
        """
        Unsubscribe from messages of a specific type.
        
        Args:
            message_type: Type of message to unsubscribe from
            handler: Callback function to remove
        """
        if handler in self._subscribers[message_type]:
            self._subscribers[message_type].remove(handler)
            logger.debug(f"Subscription removed for {message_type}")
    
    async def publish(
        self, 
        message: AgentMessage,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> None:
        """
        Publish a message to all subscribers.
        
        Args:
            message: Message to publish
            priority: Priority level for message handling
        """
        # Validate message
        if message.from_agent not in self._registered_agents:
            logger.warning(f"Message from unregistered agent: {message.from_agent}")
        
        # Add to history
        self._add_to_history(message)
        
        # Update statistics
        self._message_count[message.message_type] += 1
        
        # Log message
        logger.info(
            f"Message published: {message.message_type} from {message.from_agent} to {message.to_agent}",
            extra={
                "message_id": str(message.message_id),
                "priority": priority
            }
        )
        
        # Handle broadcast or targeted message
        if message.to_agent == "broadcast":
            await self._broadcast_message(message)
        else:
            await self._send_targeted_message(message)
    
    async def _broadcast_message(self, message: AgentMessage) -> None:
        """Broadcast message to all subscribers of the message type."""
        handlers = self._subscribers.get(message.message_type, [])
        
        # Execute handlers concurrently
        tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(message))
            else:
                # Wrap sync handlers
                tasks.append(asyncio.create_task(
                    asyncio.to_thread(handler, message)
                ))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_targeted_message(self, message: AgentMessage) -> None:
        """Send message to specific agent's queue."""
        if message.to_agent not in self._registered_agents:
            logger.warning(f"Target agent not registered: {message.to_agent}")
            return
        
        # Add to target agent's queue
        queue = self._message_queues.get(message.to_agent)
        if queue:
            await queue.put(message)
        
        # Also notify any subscribers for this message type
        await self._broadcast_message(message)
    
    async def get_messages(
        self, 
        agent_name: str, 
        timeout: Optional[float] = None
    ) -> List[AgentMessage]:
        """
        Get pending messages for an agent.
        
        Args:
            agent_name: Name of the agent
            timeout: Optional timeout in seconds
            
        Returns:
            List of pending messages
        """
        if agent_name not in self._registered_agents:
            logger.warning(f"Agent not registered: {agent_name}")
            return []
        
        queue = self._message_queues.get(agent_name)
        if not queue:
            return []
        
        messages = []
        try:
            # Get all available messages
            while True:
                if timeout is not None:
                    message = await asyncio.wait_for(
                        queue.get(), 
                        timeout=timeout
                    )
                else:
                    message = queue.get_nowait()
                messages.append(message)
        except (asyncio.QueueEmpty, asyncio.TimeoutError):
            pass
        
        return messages
    
    def _add_to_history(self, message: AgentMessage) -> None:
        """Add message to history with size limit."""
        self._message_history.append(message)
        
        # Trim history if needed
        if len(self._message_history) > self._history_limit:
            self._message_history = self._message_history[-self._history_limit:]
    
    def get_message_history(
        self, 
        message_type: Optional[str] = None,
        from_agent: Optional[str] = None,
        to_agent: Optional[str] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """
        Get message history with optional filters.
        
        Args:
            message_type: Filter by message type
            from_agent: Filter by sender
            to_agent: Filter by recipient
            limit: Maximum number of messages to return
            
        Returns:
            Filtered message history
        """
        history = self._message_history
        
        # Apply filters
        if message_type:
            history = [m for m in history if m.message_type == message_type]
        if from_agent:
            history = [m for m in history if m.from_agent == from_agent]
        if to_agent:
            history = [m for m in history if m.to_agent == to_agent]
        
        # Return most recent messages
        return history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            "registered_agents": list(self._registered_agents),
            "total_messages": sum(self._message_count.values()),
            "message_counts": dict(self._message_count),
            "history_size": len(self._message_history),
            "active_subscriptions": {
                msg_type: len(handlers) 
                for msg_type, handlers in self._subscribers.items()
            }
        }
    
    def clear_history(self) -> None:
        """Clear message history."""
        self._message_history.clear()
        logger.info("Message history cleared")
    
    async def shutdown(self) -> None:
        """Shutdown the message bus and cleanup resources."""
        logger.info("Shutting down message bus")
        
        # Clear all queues
        for queue in self._message_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        # Clear registrations
        self._registered_agents.clear()
        self._message_queues.clear()
        self._subscribers.clear()
        
        logger.info("Message bus shutdown complete")