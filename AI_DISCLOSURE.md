# AI Disclosure

## Tools Used
- Claude (Anthropic)

## How They Were Used
- Debugging MLflow model serialization errors (skops untrusted type issue, switched to pickle format).
- Troubleshooting DVC remote configuration and SSH setup.
- Clarifying technical debt categories from the Sculley et al. paper and verifying mappings to the assignment scenarios.
- Reviewing MLflow logging code.

## Impact
Claude was used as a debugging and clarification tool. All training code, MLflow runs, DVC versioning, and the reproducibility protocol were executed and verified manually. Architectural decisions (hyperparameter choices, MLP configuration, DVC workflow) were made independently.
