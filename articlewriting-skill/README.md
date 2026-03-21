# 科研写作助手 (Research Writing Assistant)

> 📄 English version: [README_EN.md](README_EN.md)

把"论文写作"从一次性聊天，升级成可追踪、可恢复、可复用的工程化协作流程。  
这个 Skill 面向本科生、研究生和早期科研人员，目标很直接：少走弯路，减少返工，把时间花在真正有价值的研究内容上。

![科研写作助手使用流程](assets/readme/workflow.png)

## 项目定位

这不是一个"只会润色句子"的提示词包，而是一套完整的科研写作协作系统。  
它会在任务开始前先对齐目标与约束，通过头脑风暴确认论文类型、研究背景、方法和章节结构，再按学科和任务路由到对应技能模块执行。

如果你在做毕业论文、课程项目论文或投稿初稿，这个 Skill 会比普通对话式写作工具更稳定，因为它强调流程、记录和回写，不依赖单轮记忆。

## 核心能力

- **头脑风暴**：7轮问答确认论文类型、学科、题目、研究背景、方法、章节结构
- **AI辅助写作**：从选题推进、正文写作、图表生成到投稿前自审，按阶段门禁执行
- **去AI化写作**：约束机械过渡词、空壳强调句、主观化表达和列表堆砌
- **学科化写作支持**：工科、社科、医学、法学模块分流
- **文献综述支持**：英文检索整合与中文文献整理协作
- **LaTeX模板支持**：用户提供学校/期刊模板，自动生成可编译的LaTeX项目
- **环境自动化**：Miniconda 安装、虚拟环境创建、绘图依赖安装与排错

## 适用平台

本 Skill 采用目录化设计，已适配以下平台：

| 平台 | 配置文件 |
|------|----------|
| Claude Code | `.claude-plugin/plugin.json` |
| Cursor | `.cursor-plugin/plugin.json` |
| Codex | `.codex/INSTALL.md` |
| OpenCode | `.opencode/INSTALL.md` |
| Gemini CLI | `GEMINI.md` |
| 其他 | `AGENTS.md` |

## 你会得到什么产物

默认情况下，Skill 产物是项目内文件，不是 Word 成品文件。

| 产物类型 | 默认格式 | 说明 |
|---|---|---|
| 写作正文 | `.md` / 纯文本 / `.tex` | 便于版本管理和后续再加工 |
| 章节文件 | `chapters/*.md` | 每章一个独立文件 |
| LaTeX项目 | `chapters/*.tex` + `main.tex` | 可直接编译 |
| 图表脚本 | `.py` | 可复现图表生成逻辑 |
| 提示词资产 | `.md` | 可复用的翻译、润色、去AI化模板 |

## 重要边界（务必先看）

1. Skill 默认不会自动生成或直接写入 `.docx`。  
2. Skill 默认不会替你"打开 Word 并排版"，需要你手动复制或用工具转换。  
3. Skill 可以生成适合粘贴进 Word 的纯文本段落，但最终样式（标题级别、页眉页脚、目录、参考文献域）需要你在 Word 端处理。  
4. 文献与数据不会编造，引用需可追溯；高风险结论请你二次核验。  

## 安装

### 方式一：直接下载

下载仓库，解压后把 `research-writing-skill/` 复制到你的论文写作目录就可以用了。

### 方式二：Git Clone

```bash
git clone https://github.com/Norman-bury/articlewriting-skill.git
cd articlewriting-skill
```

### 各平台安装

- **Codex**：参考 `.codex/INSTALL.md`
- **OpenCode**：参考 `.opencode/INSTALL.md`
- **其他平台**：将整个目录放入论文项目根目录即可

## 真实使用案例（输入 -> 输出）

![真实使用案例：输入到输出](assets/readme/real-case-input-output.png)

## 标准协作流程（推荐）

1. **头脑风暴**：说"我要写论文"，Skill会引导你确认论文类型、题目、研究背景等
2. **章节规划**：确认章节结构后，Skill在 `chapters/` 创建框架
3. **逐章写作**：按章节顺序写作，每章一个文件
4. **图表生成**：需要数据图时，Skill生成Python脚本
5. **自审检查**：使用peer-review技能进行投稿前自审
6. **交付**：手动迁移到Word/LaTeX完成最终排版

## 技能地图

| 场景 | 技能 |
|---|---|
| 入口与路由 | `skills/using-research-writing/` |
| 头脑风暴 | `skills/brainstorming-research/` |
| 章节写作 | `skills/writing-chapters/` |
| LaTeX输出 | `skills/latex-output/` |
| 通用写作规范 | `skills/writing-core/` |
| 文科/社科写作 | `skills/writing-humanities/` |
| 医学/生物写作 | `skills/writing-medical/` |
| 法学写作 | `skills/writing-law/` |
| 文献综述 | `skills/literature-review/` |
| 翻译/润色/去AI化 | `skills/prompts-collection/` |
| 投稿前自审 | `skills/peer-review/` |
| 统计分析 | `skills/statistical-analysis/` |
| Python 图表 | `skills/figures-python/` |
| 流程图/架构图 | `skills/figures-diagram/` |
| 环境安装与排错 | `skills/environment-setup/` |

## LaTeX模板使用

如果你有学校或期刊提供的LaTeX模板：

1. 将模板文件（`.cls`, `.sty`, `.tex`等）放入 `latex-templates/` 目录
2. 告诉AI"使用我的LaTeX模板"
3. AI会解析模板结构并生成对应的章节 `.tex` 文件

详细说明见 `latex-templates/README.md`

## 如何把 Markdown 交付到 Word

### 方案 A：手动复制（默认推荐）

1. 让 Skill 输出"纯文本段落版"正文（避免 Markdown 标记）
2. 在编辑器中复制正文并粘贴到 Word
3. 在 Word 中应用学校模板样式（标题、正文、图注、表注）
4. 手动检查公式、参考文献、图表编号与交叉引用

### 方案 B：Pandoc 转换（可选）

如果你本地已安装 Pandoc，可尝试：

```bash
pandoc draft.md -o draft.docx
```

说明：这只解决格式转换，不替代学校模板排版和最终人工校对。

## FAQ

### 为什么默认产物不是 Word？

因为科研协作更需要可追踪、可复用、可版本化的文本资产，Markdown 更适合过程迭代。Word 适合最终交付，所以放在最后一步处理更稳妥。

### 可以直接让我"生成最终可提交版本"吗？

可以做接近终稿的内容，但学校模板、目录域、页码、参考文献域、格式细节仍建议在 Word 端完成。

### 这个 Skill 会不会瞎编文献？

不会。规则层面明确禁止编造文献与数据；引用要求可追溯。

## 仓库结构

```text
research-writing-skill/
├── SKILL.md                    # 主入口（兼容旧平台）
├── AGENTS.md                   # 通用代理配置
├── GEMINI.md                   # Gemini CLI 配置
├── CHANGELOG.md                # 版本记录
├── .claude-plugin/             # Claude Code 配置
├── .cursor-plugin/             # Cursor 配置
├── .codex/                     # Codex 配置
├── .opencode/                  # OpenCode 配置
├── hooks/                      # 会话启动脚本
│   ├── session-start
│   ├── hooks.json
│   └── hooks-cursor.json
├── skills/                     # 技能模块目录
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
├── latex-templates/            # 用户LaTeX模板目录
├── modules/                    # 旧模块（保留兼容）
├── templates/                  # 代码模板
├── plan-template/              # 计划模板
└── scripts/                    # 工具脚本
```

## 版本

- 版本：3.0.0
- 更新日期：2026-03-19
