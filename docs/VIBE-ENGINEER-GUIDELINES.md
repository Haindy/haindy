# AI-Assisted Engineering Workflow Guide

## 1. Introduction

This guide helps developers effectively integrate AI tools, primarily Claude Code and Cursor, into everyday coding tasks. AI-assisted coding can greatly enhance productivity, but it can also lead to confusion or wasted time if not managed properly. This document helps you use AI efficiently, avoid going down unnecessary "rabbit holes," and maintain a clear, productive workflow.

### Why We Need This Document:

- To clearly define best practices and avoid common pitfalls.
- To maximize the productivity benefits of AI tools.
- To ensure consistent quality and clarity in our coding processes.

### Goals:

- Learn efficient AI-assisted coding.
- Ensure high standards for security, scalability, and architecture.
- Clearly document and share knowledge consistently.
- Avoid wasting time by going in circles or working on out-of-scope tasks.

## 2. Workflow Overview

1. Gather initial requirements.
2. Refine and structure your plan with GPT-4.5.
3. Clearly document the refined plan in markdown.
4. Have AI audit your plan for completeness.
5. Separate clearly frontend/backend tasks.
6. Split large plans into manageable segments.
7. Create a detailed design document, iterate, and approve.
8. Validate repo guidelines (CLAUDE.md).
9. Clear AI context and ensure it fully understands the plan.
10. AI summarizes next steps; validate and answer questions.
11. Use feature branches and pull requests for coding tasks.
12. Allow AI uninterrupted work and carefully review output.
13. Test locally and iterate based on feedback.
14. Merge completed PRs and update context.

## 3. Tooling

### Frontier vs. Second-Level Models

- **Frontier models** (e.g., Claude Opus 4, GPT-4.5) have larger context windows and provide better accuracy, fewer mistakes, and are better at maintaining task scope. They significantly reduce frustration by minimizing overthinking, excessive engineering, and confusion.
- **Second-level models** (e.g., Claude Sonnet 4) have smaller context windows and may often over-engineer or deviate from intended tasks.

Using frontier models increases productivity and reduces the frustration of endless revisions or misunderstood instructions.

### Tool Descriptions:

- **Claude Code**  
  A command-line (CLI) tool specifically designed for AI-assisted coding. Claude Code works entirely through a text-based interface, allowing precise and efficient context management directly within markdown files.

  **Pros:**
  - Excellent context handling and scope management
  - Highly productive workflow, ideal for structured task management
  - Reduces distractions common in traditional IDEs
  
  **Cons:**
  - Requires comfort with command-line interfaces
  - Lacks traditional visual debugging and UI conveniences found in IDEs

- **Cursor**  
  An Integrated Development Environment (IDE) based on a fork of Visual Studio Code, built explicitly for AI-assisted coding. It integrates AI coding assistance into the familiar visual and interactive environment developers are accustomed to.

  **Pros:**
  - Familiar graphical interface and IDE conveniences (e.g., visual debugging)
  - Easier onboarding for developers accustomed to VS Code
  
  **Cons:**
  - Less efficient context management compared to CLI-based tools
  - More prone to distractions and deviations from the intended task scope

**Current Recommendation:**  
The recommended tool at this time is **Claude Code**, as it provides greater productivity, clearer scope management, and more powerful task tracking than traditional IDE-based AI tools.

### Language Consideration:

AI coding tools generally perform better in English, mainly due to extensive training data available in English and the inherent English-based structure of programming languages. Using English consistently in prompts will likely yield better results and fewer misunderstandings.

## 4. Detailed Workflow Steps

### Step 1: Gather Initial Requirements

- Collect high-level requirements from brief documents, Figma mockups, and Slack conversations.
- Write clearly in your own natural language.

### Step 2: Refine and Structure Your Plan

- Engage GPT-4.5 to structure and refine your initial prompt into actionable, organized tasks.
- Discuss and clearly define each task, ensuring alignment and clarity.

### Step 3: Documentation and Repository Integration

- Use Claude Code or Cursor to save the finalized markdown plan.
- Clearly track progress within the markdown file itself.
- Use dedicated feature branches for major or breaking changes.

### Step 4: AI Audits and Recommendations

- Let Claude Code audit the markdown plan to identify gaps, improvements, or missed considerations.
- Discuss audit findings and incorporate feedback.

### Step 5: Clearly Separate Frontend/Backend Responsibilities

- Clearly delineate tasks between frontend and backend, avoiding overlap.
- Maintain distinct markdown files and separate AI sessions for each repo.

### Step 6: Manage Large Plans

- Split plans exceeding ~500 lines for easier AI and human handling.
- Clearly link all segments from a central markdown document.

### Step 7: Team Review and Iteration

- Develop a detailed design document from the finalized markdown plan.
- Iterate with team feedback and achieve approval.

### Step 8: Validate Repo Guidelines

- Check repo guidelines (CLAUDE.md) before starting coding tasks.

### Step 9: Context Management

- Regularly clear AI context (`/clear`) to prevent confusion.
- Ensure the AI has read and understands all documentation clearly.

### Step 10: AI Summarizes Next Steps

- AI clearly summarizes upcoming tasks.
- Validate AI summaries and fully address AI’s clarifying questions.

### Step 11: Coding with Feature Branches

- Let AI handle coding tasks on feature branches.
- Initiate pull requests once tasks are completed.

### Step 12: Review AI Output Carefully

- Allow AI uninterrupted coding time.
- Carefully review AI-generated summaries and code.

### Step 13: Local Testing

- Thoroughly test each implementation locally.
- Provide detailed, iterative feedback until functionality is confirmed.

### Step 14: Merge and Continue Workflow

- Review and merge completed PRs.
- Inform AI and update context clearly.
- Start the next steps again from context clearing.

## 5. Prompt Engineering Best Practices

- Clearly define the task.
- Use concise, straightforward language.
- Provide relevant examples.
- Explicitly request the desired format for output.

## 6. Quality Assurance Guidelines

- Mandatory checkpoints for human reviews:
  - Initial Plan Creation
  - Post-AI Audit
  - Final PR Review
- Follow checklists for:
  - Security considerations
  - Scalability concerns
  - Architectural guidelines

## 7. Cross-repo Management

- Maintain clear distinctions between frontend/backend tasks.
- Regularly verify alignment and consistency across repositories.

## 8. Iterative Improvement

- Hold regular (monthly or bi-monthly) retrospectives.
- Collect and apply team feedback to continually improve this workflow.

## Appendices

- Example markdown plans
- Example AI prompts and responses

---

**Use this guide to improve productivity, ensure clarity, and build great software efficiently with AI assistance.**