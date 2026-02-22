# Agent Evaluation Scorecard

Use this scorecard after each agent run. Score each criterion as:
- `0` = missing/incorrect
- `1` = partial/inconsistent
- `2` = complete/consistent

## Run Metadata

- Run ID:
- Scenario:
- Task Type (`generic` or `browser`):
- Evaluator:
- Timestamp (UTC):

## Criteria (0-2 each)

| Criterion | Score | Notes |
|---|---:|---|
| Goal specification |  |  |
| Plan quality |  |  |
| Tool-call correctness |  |  |
| Postcondition checks |  |  |
| Retry discipline |  |  |
| State/memory use |  |  |
| Verification depth |  |  |
| Error attribution |  |  |
| Completion evidence |  |  |
| Communication quality |  |  |

## Totals

- Total score (max 20):
- Band:
  - `0-8`: Unreliable
  - `9-14`: Usable but brittle
  - `15-17`: Strong
  - `18-20`: Production-grade

## Browser Gate (required for browser tasks)

For `browser` tasks, these must all be `2`:
- Postcondition checks
- Verification depth
- Retry discipline

If any are below `2`, mark the run as failed regardless of total score.

## Debrief

- Strengths:
- Failure modes:
- Top 3 fixes before next run:
