# Agent Prompt

This is `opt-SPTnet`, an optimized, packaged reimplementation and critical
reproduction of SPTnet for an MPhil thesis. The deliverable is the 7000-word
report in `report/` (`report/main.tex`). The reference/original codebase is the
sibling repository `../SPTnet`.

Before working, read the files in `context/`. They are the project's memory and
the single source of truth for intent, status, and history — prefer them over
re-deriving context from the code or git log.

## The context files

- `context/context.md` — **what the project is and what the report argues.**
  Project purpose, thesis framing, the four project aims, and (importantly) the
  current report focus: which contributions are the headline versus secondary.
  Read this first to understand the frame before touching anything. It changes
  rarely — only when the project's direction or thesis narrative changes.

- `context/changes.md` — **what was changed in the package versus `../SPTnet`.**
  A catalogue of the refactor and optimizations: package structure, public API,
  model/loss/training changes, CRLB handling, the Python data generator,
  inference, segmentation/stitching, visualization, tests, and docs. This is the
  evidence base for the report's packaging/optimization/reproducibility claims.
  It documents reusable package behaviour, NOT one-off experiment scripts.

- `context/plan.md` — **what to do next and why.** Current phase, the ordered
  task list, the main-tasks summary, and open questions. This is the file to
  consult to decide what to work on, and the one most likely to be edited by the
  user between sessions — treat its latest state as authoritative even if it
  conflicts with older notes.

- `context/notes.md` — **the durable, dated project log.** Chronological findings,
  decisions, and the reasoning behind them from previous agents (e.g. the
  evaluation audit, the matched per-track evaluation, the fine-tune and
  forgetting tests, the benchmark protocol). Read this to understand HOW the
  current plan was arrived at and to avoid repeating settled analysis. Entries
  are durable knowledge, not transient chatter; convert relative dates to
  absolute when adding.

- `context/style.md` - a writing style guide used to make sure suggested passages of text match the existing report. Follow the reccomendations of the style guide whilst still producing high quality writing.

- `context/prompt.md` — this file: the entry point describing the context files
  and the working discipline.

## Keeping the context files current

- Add new durable findings, results, and the reasoning behind decisions to
  `context/notes.md` (dated).
- Add or revise next steps and open questions in `context/plan.md`; keep its
  "main tasks" summary in sync with reality.
- Update `context/changes.md` when core package functionality or optimization
  behaviour changes.
- Update `context/context.md` only when the project framing, aims, or report
  focus change.
- Keep volatile specifics (exact numbers, current priorities) in
  `notes.md`/`plan.md`; keep `context.md` and this file about stable framing and
  structure so they do not go stale.

## Working discipline

Preserve the project goal: reproducible, extensible SPTnet functionality
supporting the report in `report/`. Prefer changes that make workflows easier to
reproduce, test, explain, and maintain. Preserve the distinction between core
package functionality (`src/sptnet/`) and experiment-specific scripts
(`experiments/`, one-off SLURM variants, notebook analysis). Every report
figure/table should trace to a committed source artifact or a notebook cell.
