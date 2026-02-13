# Project Guidelines

## Code Style
- Never use inline imports (imports inside functions). All imports go at the top of the file.

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update 'tasks/lessons.md' with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests -> then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management
1. **Plan First**: Write plan to 'tasks/todo.md' with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review to 'tasks/todo.md'
6. **Capture Lessons**: Update 'tasks/lessons.md' after corrections

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

## Exploration & Learning

### Option Exploration
- **Branch Before Commit**: When multiple approaches exist, sketch 2-3 options with tradeoffs before implementing
- **Timebox Exploration**: Set a mental limit on research time, then pick and move
- **Prototype Fast**: Quick throwaway code to validate assumptions beats theorizing
- **Kill Bad Ideas Early**: If an approach smells wrong after 10 mins, pivot immediately

### Learning from Codebase
- **Pattern Mining**: Before writing new code, find 3 similar implementations in the codebase and follow the established pattern
- **Ask Why**: When you see unfamiliar patterns, understand the reasoning before changing them
- **Document Discoveries**: Non-obvious codebase quirks go in README or inline comments where found

### Decision Framework
1. Does this solution already exist somewhere? (search first, write second)
2. What's the simplest thing that could work?
3. What breaks if I'm wrong?
4. Can I easily reverse this decision later?

### Knowledge Capture
- After solving a hard problem: distill the insight into `tasks/lessons.md`
- Tag lessons by domain: `[perf]`, `[arch]`, `[gotcha]`, `[pattern]`
- Review relevant lessons before touching unfamiliar code areas
- Delete lessons that become obsolete or obvious