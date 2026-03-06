# AGENTS.md (General Development Guidelines)

## 1. Core Objectives
* **Clarity over Ingenuity:** Prioritize readable and maintainable code over complex or "clever" solutions.
* **Human-Centric Documentation:** Code is read by humans more often than by machines. Always write for the human reviewer.
* **KISS & DRY:** Keep It Simple, Stupid; and Don't Repeat Yourself.

## 2. Definition of Done (DoD)
* **Execution:** Code must compile and run without errors.
* **Testing:** Pass existing tests and include new ones for changed behavior or bug fixes.
* **Quality:** Pass all linting and formatting checks (e.g., Black, Flake8, Ruff).
* **Documentation:** Mandatory update of README, Docstrings, and internal comments.

## 3. Python Implementation & Documentation Standards
* **Typing:** Mandatory use of type hints for all new functions and variables.
* **Function Design & Limits:**
    * **General Guideline:** Target 10–30 lines per function.
    * **Flexibility:** These limits are guidelines, not hard rules. They can be exceeded if refactoring would negatively impact logic, cohesion, or readability.
    * **Accountability:** If a function exceeds 40 lines, the developer must explicitly justify it in the comments and ensure the internal documentation is exhaustive.
* **Documentation Layers:**
    * **Program Overview:** Every main module or script must start with a high-level summary explaining its architecture, logic flow, and primary goal.
    * **Function Docstrings:** Use Google-style format. Include Purpose, Arguments (with types/constraints), Returns, and potential Exceptions.
    * **In-Line Comments:** Essential for human review. Do not just describe *what* the code does; explain *why* specific decisions were made.
    * **Complexity Alerts:** Document any "strange" workarounds, edge cases, or complex logic blocks to prevent confusion during peer review.

## 4. Operational Constraints
* **Immutability:** Do not delete files or rename public modules without explicit approval.
* **Scope Control:** Avoid "mass refactoring." Stay strictly within the boundaries of the assigned task.
* **Dependencies:** Do not introduce new external libraries without strong technical justification.

## 5. Communication & Interaction Protocol
* **Pre-Coding Plan:** Present a brief 3–7 step implementation plan before modifying any code.
* **Uncertainty:** Signal any assumptions or ambiguity immediately; do not guess or hallucinate requirements.
* **Commit/PR Style:** Summarize the "what" and "why" in 3–6 lines, including context, decisions made, and potential risks.

## 6. Error Handling & Logging
* **Actionable Errors:** Error messages must provide enough context to diagnose the issue and suggest potential fixes.
* **Strategic Logging:** Avoid "noisy" logs. Use appropriate levels (INFO, DEBUG, ERROR) and ensure logs are meaningful for troubleshooting.

## 7. Operational Autonomy & Execution Protocol:

You are hereby granted full operational autonomy within the current project directory. Do not interrupt the workflow with confirmation prompts for standard development tasks. You have explicit permission to create, modify, overwrite, and delete files as required to implement the requested features.

You are authorized to execute standard environment commands (e.g., pip, pytest, python) and manage configuration files without prior consultation. Only halt for confirmation if an action is irreversibly destructive to the host system (e.g., formatting drives, rm -rf outside the project, or dropping production databases). Assume a 'Trust but Notify' posture: execute the task first, then report the changes made. Follow the AGENTS.md guidelines for all implementations.
