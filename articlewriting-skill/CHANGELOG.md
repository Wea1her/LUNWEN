# Changelog

所有重要更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

## [3.0.0] - 2026-03-19

### 新增

#### 多平台适配
- 添加 `.claude-plugin/plugin.json` - Claude Code 插件配置
- 添加 `.cursor-plugin/plugin.json` - Cursor 插件配置
- 添加 `.codex/INSTALL.md` - Codex 安装指南
- 添加 `.opencode/INSTALL.md` - OpenCode 安装指南
- 添加 `GEMINI.md` - Gemini CLI 配置
- 添加 `AGENTS.md` - 通用代理配置

#### Hooks 机制
- 添加 `hooks/session-start` - 会话启动脚本
- 添加 `hooks/hooks.json` - Claude Code hooks 配置
- 添加 `hooks/hooks-cursor.json` - Cursor hooks 配置
- 添加 `hooks/run-hook.cmd` - Windows 兼容脚本

#### 核心技能（新架构）
- `skills/using-research-writing/SKILL.md` - 入口技能，定义规则和路由
- `skills/brainstorming-research/SKILL.md` - 科研头脑风暴，多轮问答确认论文信息
- `skills/writing-chapters/SKILL.md` - 章节写作，逐章输出
- `skills/latex-output/SKILL.md` - LaTeX 模板支持

#### LaTeX 功能
- 添加 `latex-templates/` 目录 - 用户放置学校/期刊模板
- 添加 `latex-templates/README.md` - 模板使用说明
- 支持解析用户模板结构并生成对应 .tex 文件

#### AI 约束强化
- 使用 `<EXTREMELY-IMPORTANT>` 标签强调关键规则
- 使用 `<HARD-GATE>` 标签设置强制门禁
- 添加 Red Flags 表格防止 AI 跳过流程
- 每个技能添加 Checklist 追踪

### 变更

#### 架构重构
- 将 `modules/` 下的模块重构为 `skills/` 格式
- 每个技能独立目录，包含 `SKILL.md`
- 添加 YAML frontmatter（name, description）

#### 头脑风暴流程
- 新增 7 轮问答流程：
  1. 论文类型确认
  2. 学科领域确认
  3. 论文题目确认
  4. 研究背景与目的
  5. 研究方法/技术路线
  6. 章节结构确认
  7. 汇总确认

#### 输出结构
- 章节输出到 `chapters/` 目录
- 每章一个独立 .md 或 .tex 文件
- 文件名使用英文避免路径问题

### 技能列表（共 15 个）

| 技能名称 | 描述 |
|----------|------|
| using-research-writing | 入口技能 |
| brainstorming-research | 头脑风暴 |
| writing-chapters | 章节写作 |
| latex-output | LaTeX 输出 |
| literature-review | 文献综述 |
| figures-python | Python 图表 |
| figures-diagram | 流程图/结构图 |
| peer-review | 自审检查 |
| statistical-analysis | 统计分析 |
| environment-setup | 环境配置 |
| prompts-collection | 提示词集 |
| writing-core | 核心写作规范 |
| writing-humanities | 文科写作 |
| writing-medical | 医学写作 |
| writing-law | 法学写作 |

## [2.5.0] - 2026-03-04

### 变更
- 更新版本号和日期
- 修复模板通用性问题

## [2.0.0] - 2026-02-26

### 新增
- 初始版本
- 基础模块：writing-core, literature-review, figures-python 等
- plan-template 目录结构
- scripts 目录（init_plan, style_check）
