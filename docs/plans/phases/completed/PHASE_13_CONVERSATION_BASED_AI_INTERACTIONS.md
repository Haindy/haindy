# Phase 13 — Conversation-Based AI Interactions

## Overview
Implement conversation threads for Action Agent to maintain context across AI interactions within the same action.

**ETA**: 3-4 days

## Status: ✅ Completed

### Problem Statement
Currently, each AI call is isolated without context from previous interactions. This leads to:
- Redundant information in every prompt
- Inability to reference previous analyses
- Complex workarounds for image comparison
- GPT-4o-mini limitations with multiple images

### Proposed Solution
Implement conversation threads for Action Agent where:
- **One conversation per action** - New action = new conversation
- AI remembers previous screenshots and analyses within the same action
- Natural comparison between states (before/after)
- Single image per message (works with GPT-4o-mini)
- Errors and retries remain part of the conversation

### Implementation Plan

#### 1. Conversation State Management (1 day)
- Add `conversation_history` to ActionAgent
- Implement message history tracking per action (not per step)
- Token-based sliding window with raw message storage (no compression)
- Reset conversation when moving to next action

#### 2. Replace AI Interaction Code (1-2 days)
- **Replace** (not wrap) existing `call_openai_with_debug` implementation
- Build conversation-aware OpenAI calls from scratch
- Include all messages in conversation (including errors/retries)
- Implement token-based context window management

#### 3. Optimize Prompts (1 day)
- Rewrite prompts to leverage conversation context
- Remove redundant information
- Implement natural language transitions between attempts

#### 4. Testing & Validation (1 day)
- Test with Wikipedia scenario
- Verify improved accuracy
- Measure token usage optimization

### Scope
- **Action Agent only** - Other agents will be addressed in future phases if needed
- No wrappers or fallbacks - direct replacement of existing code
- Conversation boundary: Start of action → End of action (success or final failure)

### Success Criteria
- AI accurately references previous screenshots within an action
- Successful visual comparison without sending multiple images
- Improved typing detection accuracy
- Reduced token usage by 30%+
- Clean conversation reset between actions

### Benefits
1. **Improved Accuracy**: AI can compare before/after states naturally
2. **Token Efficiency**: Avoid repeating context in every call
3. **Better Error Recovery**: AI understands what was already tried
4. **Natural Interaction**: More human-like conversation flow
5. **GPT-4o-mini Compatibility**: Works within single-image limitations