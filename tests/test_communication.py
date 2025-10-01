"""
Tests for the inter-agent communication system.
"""

import asyncio
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.core.types import AgentMessage
from src.orchestration.communication import MessageBus, MessageType, MessagePriority


@pytest.fixture
def message_bus():
    """Create a MessageBus instance for testing."""
    return MessageBus()


@pytest.fixture
def sample_message():
    """Create a sample message for testing."""
    return AgentMessage(
        from_agent="test_agent_1",
        to_agent="test_agent_2",
        message_type=MessageType.EXECUTE_STEP,
        content={"step": "test_step"}
    )


class TestMessageBus:
    """Test cases for MessageBus."""
    
    def test_register_agent(self, message_bus):
        """Test agent registration."""
        message_bus.register_agent("test_agent")
        
        assert "test_agent" in message_bus._registered_agents
        assert "test_agent" in message_bus._message_queues
        
        # Test duplicate registration
        message_bus.register_agent("test_agent")  # Should not raise
    
    def test_unregister_agent(self, message_bus):
        """Test agent unregistration."""
        message_bus.register_agent("test_agent")
        message_bus.unregister_agent("test_agent")
        
        assert "test_agent" not in message_bus._registered_agents
        assert "test_agent" not in message_bus._message_queues
        
        # Test unregistering non-existent agent
        message_bus.unregister_agent("non_existent")  # Should not raise
    
    def test_subscribe_unsubscribe(self, message_bus):
        """Test message subscription and unsubscription."""
        handler = Mock()
        
        # Subscribe
        message_bus.subscribe(MessageType.EXECUTE_STEP, handler, "test_agent")
        assert handler in message_bus._subscribers[MessageType.EXECUTE_STEP]
        
        # Unsubscribe
        message_bus.unsubscribe(MessageType.EXECUTE_STEP, handler)
        assert handler not in message_bus._subscribers[MessageType.EXECUTE_STEP]
    
    @pytest.mark.asyncio
    async def test_publish_broadcast(self, message_bus):
        """Test broadcasting messages."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        
        message_bus.subscribe(MessageType.EXECUTE_STEP, handler1)
        message_bus.subscribe(MessageType.EXECUTE_STEP, handler2)
        
        # Register sender
        message_bus.register_agent("test_agent_1")
        
        # Broadcast message
        message = AgentMessage(
            from_agent="test_agent_1",
            to_agent="broadcast",
            message_type=MessageType.EXECUTE_STEP,
            content={"test": "data"}
        )
        
        await message_bus.publish(message)
        
        # Verify handlers were called
        handler1.assert_called_once_with(message)
        handler2.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_publish_targeted(self, message_bus):
        """Test targeted message delivery."""
        message_bus.register_agent("sender")
        message_bus.register_agent("receiver")
        
        message = AgentMessage(
            from_agent="sender",
            to_agent="receiver",
            message_type=MessageType.EXECUTE_STEP,
            content={"test": "data"}
        )
        
        await message_bus.publish(message)
        
        # Check message in receiver's queue
        messages = await message_bus.get_messages("receiver")
        assert len(messages) == 1
        assert messages[0] == message
    
    @pytest.mark.asyncio
    async def test_publish_with_priority(self, message_bus, sample_message):
        """Test message publishing with priority."""
        message_bus.register_agent("test_agent_1")
        
        # Publish with different priorities
        await message_bus.publish(sample_message, MessagePriority.HIGH)
        
        # Check message in history
        assert len(message_bus._message_history) == 1
        assert message_bus._message_count[MessageType.EXECUTE_STEP] == 1
    
    @pytest.mark.asyncio
    async def test_get_messages_timeout(self, message_bus):
        """Test getting messages with timeout."""
        message_bus.register_agent("test_agent")
        
        # Try to get messages with short timeout
        messages = await message_bus.get_messages("test_agent", timeout=0.1)
        assert messages == []
    
    def test_message_history(self, message_bus, sample_message):
        """Test message history functionality."""
        message_bus.register_agent("test_agent_1")
        
        # Add messages to history
        asyncio.run(message_bus.publish(sample_message))
        
        # Get history
        history = message_bus.get_message_history()
        assert len(history) == 1
        assert history[0] == sample_message
        
        # Test filtered history
        history = message_bus.get_message_history(
            message_type=MessageType.EXECUTE_STEP
        )
        assert len(history) == 1
        
        history = message_bus.get_message_history(
            from_agent="test_agent_1"
        )
        assert len(history) == 1
        
        history = message_bus.get_message_history(
            to_agent="test_agent_2"
        )
        assert len(history) == 1
    
    def test_statistics(self, message_bus, sample_message):
        """Test message bus statistics."""
        message_bus.register_agent("test_agent_1")
        message_bus.register_agent("test_agent_2")
        
        asyncio.run(message_bus.publish(sample_message))
        
        stats = message_bus.get_statistics()
        
        assert "test_agent_1" in stats["registered_agents"]
        assert "test_agent_2" in stats["registered_agents"]
        assert stats["total_messages"] == 1
        assert stats["message_counts"][MessageType.EXECUTE_STEP] == 1
        assert stats["history_size"] == 1
    
    def test_clear_history(self, message_bus, sample_message):
        """Test clearing message history."""
        message_bus.register_agent("test_agent_1")
        asyncio.run(message_bus.publish(sample_message))
        
        assert len(message_bus._message_history) == 1
        
        message_bus.clear_history()
        assert len(message_bus._message_history) == 0
    
    @pytest.mark.asyncio
    async def test_sync_handler(self, message_bus):
        """Test synchronous message handler."""
        sync_handler = Mock()
        
        message_bus.subscribe(MessageType.INFO, sync_handler)
        message_bus.register_agent("sender")
        
        message = AgentMessage(
            from_agent="sender",
            to_agent="broadcast",
            message_type=MessageType.INFO,
            content={"info": "test"}
        )
        
        await message_bus.publish(message)
        
        # Give time for async wrapper
        await asyncio.sleep(0.1)
        
        sync_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_message_history_limit(self, message_bus):
        """Test message history size limit."""
        message_bus._history_limit = 5
        message_bus.register_agent("sender")
        
        # Add more messages than limit
        for i in range(10):
            message = AgentMessage(
                from_agent="sender",
                to_agent="broadcast",
                message_type=MessageType.INFO,
                content={"index": i}
            )
            await message_bus.publish(message)
        
        # History should be limited
        assert len(message_bus._message_history) == 5
        # Should have most recent messages
        assert message_bus._message_history[-1].content["index"] == 9
    
    @pytest.mark.asyncio
    async def test_unregistered_sender_warning(self, message_bus):
        """Test warning for unregistered sender."""
        message = AgentMessage(
            from_agent="unregistered",
            to_agent="broadcast",
            message_type=MessageType.INFO,
            content={}
        )
        
        # Should not raise, just warn
        await message_bus.publish(message)
    
    @pytest.mark.asyncio
    async def test_targeted_to_unregistered(self, message_bus):
        """Test sending to unregistered agent."""
        message_bus.register_agent("sender")
        
        message = AgentMessage(
            from_agent="sender",
            to_agent="unregistered",
            message_type=MessageType.INFO,
            content={}
        )
        
        # Should not raise, just warn
        await message_bus.publish(message)
    
    @pytest.mark.asyncio
    async def test_shutdown(self, message_bus):
        """Test message bus shutdown."""
        message_bus.register_agent("agent1")
        message_bus.register_agent("agent2")
        
        handler = Mock()
        message_bus.subscribe(MessageType.INFO, handler)
        
        await message_bus.shutdown()
        
        assert len(message_bus._registered_agents) == 0
        assert len(message_bus._message_queues) == 0
        assert len(message_bus._subscribers) == 0


class TestMessageTypes:
    """Test message type enums."""
    
    def test_message_types_defined(self):
        """Test all message types are defined."""
        expected_types = [
            "START_TEST", "STOP_TEST", "PAUSE_TEST", "RESUME_TEST",
            "EXECUTE_STEP", "DETERMINE_ACTION", "PLAN_TEST",
            "STEP_COMPLETED", "STEP_FAILED",
            "ACTION_DETERMINED", "PLAN_CREATED",
            "STATUS_UPDATE", "ERROR", "WARNING", "INFO"
        ]
        
        for msg_type in expected_types:
            assert hasattr(MessageType, msg_type)


class TestMessagePriority:
    """Test message priority enum."""
    
    def test_priority_levels(self):
        """Test all priority levels are defined."""
        assert MessagePriority.CRITICAL == "critical"
        assert MessagePriority.HIGH == "high"
        assert MessagePriority.NORMAL == "normal"
        assert MessagePriority.LOW == "low"