# AI Disclosure

## Tools Used
- **Claude (Anthropic)** — conversational AI assistant

## How They Were Used
- Used Claude to debug MLflow model serialization errors (skops untrusted type issue → switched to pickle format).
- Used Claude to troubleshoot DVC remote configuration and SSH setup for data versioning.
- Used Claude to clarify technical debt categories from the Sculley et al. paper and verify mappings to assignment scenarios.
- Used Claude to review and suggest improvements to the MLflow logging code structure.

## Impact
Claude was used as a debugging and conceptual clarification tool. All training code, MLflow experiment runs, DVC versioning steps, and the reproducibility protocol were executed and verified manually. The architectural decisions (hyperparameter choices, MLP configuration, DVC workflow) were made independently.
