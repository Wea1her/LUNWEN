# Research Writing Assistant

Upgrade "paper writing" from one-off chat sessions into a trackable, recoverable, and reusable engineering-style collaboration workflow.  
This Skill is designed for undergraduates, graduate students, and early-stage researchers with a clear goal: fewer detours, less rework, and more time spent on research that truly matters.

![Research Writing Assistant Workflow](assets/readme/workflow.png)

## What This Is

This is not a prompt pack that only polishes sentences. It is a complete research writing collaboration system.  
Before starting any task, it aligns goals and constraints through a brainstorming process to confirm paper type, research background, methods, and chapter structure, then routes to the appropriate skill module based on your discipline and task type.

If you are working on a thesis, a course project paper, or a submission draft, this Skill is more reliable than ordinary conversational writing tools — because it emphasizes process, documentation, and write-back, without depending on single-session memory.

## Core Capabilities

- **Brainstorming**: 7-round Q&A to confirm paper type, discipline, title, research background, methods, chapter structure
- **End-to-end collaboration**: From topic development, body writing, and figure generation to pre-submission self-review, executed with stage gates
- **De-AI writing**: Constrains mechanical transition words, hollow emphasis phrases, subjective expressions, and bullet-point dumping
- **Discipline-specific writing support**: Routed modules for engineering, social sciences, medicine, and law
- **Literature review support**: Integration of English-language search and Chinese literature organization
- **LaTeX template support**: Provide your institution/journal template, auto-generate compilable LaTeX project
- **Environment automation**: Miniconda installation, virtual environment creation, plotting dependency setup

## Supported Platforms

This Skill uses a directory-based design, adapted for the following platforms:

| Platform | Configuration |
|----------|---------------|
| Claude Code | `.claude-plugin/plugin.json` |
| Cursor | `.cursor-plugin/plugin.json` |
| Codex | `.codex/INSTALL.md` |
| OpenCode | `.opencode/INSTALL.md` |
| Gemini CLI | `GEMINI.md` |
| Others | `AGENTS.md` |

## What You Get

By default, Skill outputs are project files — not finished Word documents.

| Output Type | Default Format | Notes |
|---|---|---|
| Written body | `.md` / plain text / `.tex` | Suitable for version control and further processing |
| Chapter files | `chapters/*.md` | One file per chapter |
| LaTeX project | `chapters/*.tex` + `main.tex` | Directly compilable |
| Figure scripts | `.py` | Reproducible figure generation logic |
| Prompt assets | `.md` | Reusable templates for translation, polishing, and de-AI-ification |

## Important Boundaries (Read First)

1. The Skill does not automatically generate or write directly to `.docx` files by default.  
2. The Skill does not open Word and apply formatting on your behalf — you will need to copy manually or use a conversion tool.  
3. The Skill can generate plain-text paragraphs ready to paste into Word, but final styling (heading levels, headers/footers, table of contents, reference fields) must be handled in Word.  
4. References and data are never fabricated; all citations must be traceable. Please independently verify high-risk conclusions.

## Installation

### Option 1: Direct Download

Download the repository, extract it, and copy `research-writing-skill/` into your paper writing directory.

### Option 2: Git Clone

```bash
git clone https://github.com/Norman-bury/articlewriting-skill.git
cd articlewriting-skill
```

### Platform-specific Installation

- **Codex**: See `.codex/INSTALL.md`
- **OpenCode**: See `.opencode/INSTALL.md`
- **Others**: Place the entire directory in your paper project root

## Real Usage Example (Input → Output)

![Real Usage Example: Input to Output](assets/readme/real-case-input-output.png)

## Standard Collaboration Workflow (Recommended)

1. **Brainstorming**: Say "I want to write a paper", the Skill will guide you to confirm paper type, title, research background, etc.
2. **Chapter planning**: After confirming chapter structure, the Skill creates framework in `chapters/`
3. **Chapter writing**: Write chapter by chapter, one file per chapter
4. **Figure generation**: When data figures are needed, the Skill generates Python scripts
5. **Self-review**: Use the peer-review skill for pre-submission review
6. **Delivery**: Manually migrate to Word/LaTeX for final formatting

## Skill Map

| Scenario | Skill |
|---|---|
| Entry and routing | `skills/using-research-writing/` |
| Brainstorming | `skills/brainstorming-research/` |
| Chapter writing | `skills/writing-chapters/` |
| LaTeX output | `skills/latex-output/` |
| General writing standards | `skills/writing-core/` |
| Humanities / social science writing | `skills/writing-humanities/` |
| Medical / biology writing | `skills/writing-medical/` |
| Law writing | `skills/writing-law/` |
| Literature review | `skills/literature-review/` |
| Translation / polishing / de-AI | `skills/prompts-collection/` |
| Pre-submission self-review | `skills/peer-review/` |
| Statistical analysis | `skills/statistical-analysis/` |
| Python figures | `skills/figures-python/` |
| Flowcharts / architecture diagrams | `skills/figures-diagram/` |
| Environment setup and troubleshooting | `skills/environment-setup/` |

## LaTeX Template Usage

If you have a LaTeX template from your institution or journal:

1. Place template files (`.cls`, `.sty`, `.tex`, etc.) in `latex-templates/` directory
2. Tell the AI "use my LaTeX template"
3. The AI will parse template structure and generate corresponding `.tex` chapter files

See `latex-templates/README.md` for details.

## Delivering Markdown to Word

### Option A: Manual Copy (Default Recommendation)

1. Ask the Skill to output a "plain-text paragraph version" (avoiding Markdown markers)
2. Copy the body in your editor and paste it into Word
3. Apply your institution's template styles in Word (headings, body text, captions)
4. Manually check equations, references, figure/table numbers, and cross-references

### Option B: Pandoc Conversion (Optional)

If you have Pandoc installed locally:

```bash
pandoc draft.md -o draft.docx
```

Note: This only handles format conversion — it does not replace your institution's template styling or final manual review.

## FAQ

### Why are the default outputs not Word files?

Research collaboration needs text assets that are trackable, reusable, and version-controlled. Markdown is better suited for iterative work. Word is appropriate for final delivery, so it is handled as the last step.

### Can it generate the final submittable version for me?

It can produce content close to a final draft, but your institution's template, table of contents fields, page numbers, reference fields, and formatting details should still be finalized in Word.

### Will this Skill fabricate references?

No. The rules explicitly prohibit fabricating references or data. All citations must be traceable.

## Repository Structure

```text
research-writing-skill/
├── SKILL.md                    # Main entry (legacy platform compatible)
├── AGENTS.md                   # General agent configuration
├── GEMINI.md                   # Gemini CLI configuration
├── CHANGELOG.md                # Version history
├── .claude-plugin/             # Claude Code configuration
├── .cursor-plugin/             # Cursor configuration
├── .codex/                     # Codex configuration
├── .opencode/                  # OpenCode configuration
├── hooks/                      # Session start scripts
│   ├── session-start
│   ├── hooks.json
│   └── hooks-cursor.json
├── skills/                     # Skill modules directory
│   ├── using-research-writing/
│   ├── brainstorming-research/
│   ├── writing-chapters/
│   ├── latex-output/
│   ├── literature-review/
│   ├── figures-python/
│   ├── figures-diagram/
│   ├── peer-review/
│   ├── statistical-analysis/
│   ├── environment-setup/
│   ├── prompts-collection/
│   ├── writing-core/
│   ├── writing-humanities/
│   ├── writing-medical/
│   └── writing-law/
├── latex-templates/            # User LaTeX templates directory
├── modules/                    # Legacy modules (kept for compatibility)
├── templates/                  # Code templates
├── plan-template/              # Plan templates
└── scripts/                    # Utility scripts
```

## Version

- Version: 3.0.0
- Updated: 2026-03-19
- Maintenance goal: Stable workflow, traceable content, deliverable outputs, multi-platform support
