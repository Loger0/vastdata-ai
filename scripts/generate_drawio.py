#!/usr/bin/env python3
"""Generate a DrawIO architecture diagram file from Multica system analysis."""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import datetime

# ── Constants ──────────────────────────────────────────────
STYLES = {
    "layer_box": "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=13;fontStyle=1;verticalAlign=top;",
    "layer_box_green": "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=13;fontStyle=1;verticalAlign=top;",
    "layer_box_orange": "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;fontSize=13;fontStyle=1;verticalAlign=top;",
    "layer_box_purple": "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=13;fontStyle=1;verticalAlign=top;",
    "layer_box_red": "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=13;fontStyle=1;verticalAlign=top;",
    "layer_box_cyan": "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#007FFF;fontSize=13;fontStyle=1;verticalAlign=top;",
    "layer_box_blue": "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=13;fontStyle=1;verticalAlign=top;",
    "component": "rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontSize=11;",
    "component_blue": "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=11;",
    "component_green": "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=11;",
    "component_orange": "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;fontSize=11;",
    "component_red": "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=11;",
    "component_yellow": "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;",
    "arrow": "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#333333;strokeWidth=2;endArrow=block;endFill=1;",
    "dashed_arrow": "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#666666;strokeWidth=1;endArrow=block;endFill=1;dashed=1;",
    "title": "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=20;fontStyle=1;",
    "subtitle": "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=14;fontStyle=1;",
    "label": "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=10;",
    "start_end": "rounded=1;whiteSpace=wrap;html=1;fillColor=#60C060;strokeColor=#147850;fontSize=12;fontStyle=1;",
    "decision": "rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;",
    "gate_red": "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=11;fontStyle=1;",
    "block": "rounded=1;whiteSpace=wrap;html=1;fillColor=#000000;strokeColor=#333333;fontColor=#FF4444;fontSize=12;fontStyle=1;",
    "agent_box": "rounded=1;whiteSpace=wrap;html=1;fillColor=#E6F2FF;strokeColor=#4D90FE;fontSize=10;",
    "skill_box": "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF9E6;strokeColor=#E6C300;fontSize=10;",
}

PAGE_W = 1600
PAGE_H = 1200
CELL_ID = [2]  # mutable counter


def next_id():
    CELL_ID[0] += 1
    return str(CELL_ID[0])


def make_root():
    root = ET.Element("root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")
    return root


def add_vertex(root, parent, x, y, w, h, style, value, vid=None):
    if vid is None:
        vid = next_id()
    cell = ET.SubElement(
        root,
        "mxCell",
        id=vid,
        value=value,
        style=style,
        vertex="1",
        parent=parent,
    )
    ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
    return vid


def add_edge(root, parent, source, target, style="", value="", eid=None):
    if eid is None:
        eid = next_id()
    attrs = {
        "id": eid,
        "value": value,
        "style": style,
        "edge": "1",
        "parent": parent,
        "source": source,
        "target": target,
    }
    cell = ET.SubElement(root, "mxCell", attrs)
    ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
    return eid


def add_layer_box(root, parent, x, y, w, h, title, style_key="layer_box"):
    return add_vertex(root, parent, x, y, w, h, STYLES[style_key], title)


def add_component(root, parent, x, y, w, h, label, style_key="component"):
    return add_vertex(root, parent, x, y, w, h, STYLES[style_key], label)


def build_diagram(name, root, page_w=PAGE_W, page_h=PAGE_H):
    mxgm = ET.Element(
        "mxGraphModel",
        dx="1422",
        dy="794",
        grid="1",
        gridSize="10",
        guides="1",
        tooltips="1",
        connect="1",
        arrows="1",
        fold="1",
        page="1",
        pageScale="1",
        pageWidth=str(page_w),
        pageHeight=str(page_h),
        math="0",
        shadow="0",
    )
    mxgm.append(root)
    diagram = ET.Element("diagram", id=f"diag_{name}", name=name)
    diagram.append(mxgm)
    return diagram


# ═══════════════════════════════════════════════════════════
# PAGE 1: 系统全景图
# ═══════════════════════════════════════════════════════════
def build_page1():
    root = make_root()
    parent = "1"

    # ── Title ──
    add_vertex(root, parent, 550, 10, 500, 40, STYLES["title"], "Multica 生态适配系统 — 全景架构图 (v5)")

    # ── Layer 1: 用户层 ──
    add_layer_box(root, parent, 20, 60, 500, 90, "👤 用户层", "layer_box")
    add_component(root, parent, 40, 95, 140, 40, "Claude Code CLI", "component_blue")
    add_component(root, parent, 200, 95, 140, 40, "VSCode Extension", "component_blue")
    add_component(root, parent, 360, 95, 140, 40, "Multica Web", "component_blue")

    # ── Layer 2: 本地 Skills 层 ──
    add_layer_box(root, parent, 20, 170, 500, 120, "🧩 本地 Skills 层", "layer_box_green")
    skills = [
        ("creating-multica-issues", 30), ("multica-eco-adapter-ops", 195),
        ("vastbase-python-sdk", 360), ("vexdb", 30),
        ("vastbase", 160), ("tapd-openapi", 270),
        ("ragflow-dataset-ingest", 380),
    ]
    y_row1 = 205
    y_row2 = 245
    for i, (name, x) in enumerate(skills):
        row_y = y_row1 if i < 3 else y_row2
        add_component(root, parent, x + 10, row_y, 155 if i < 3 else 105, 30, name, "component_green")

    # ── Layer 2b: Memory ──
    add_layer_box(root, parent, 540, 170, 250, 120, "🧠 Memory 系统", "layer_box_purple")
    add_component(root, parent, 555, 210, 220, 30, "integration-test-mandatory-gate", "component")
    add_component(root, parent, 555, 250, 220, 30, "multica-v5-convention-extractor", "component")

    # ── Layer 3: 配置层 ──
    add_layer_box(root, parent, 20, 310, 770, 100, "⚙️ 配置层", "layer_box_orange")
    add_component(root, parent, 35, 350, 220, 45, "settings.json\nmodel routing · effortLevel: xhigh", "component_orange")
    add_component(root, parent, 270, 350, 240, 45, "settings.local.json\npermissions allowlist", "component_orange")
    add_component(root, parent, 525, 350, 250, 45, "PreToolUse Hooks\nTAPD 关键词自动路由", "component_orange")

    # ── Layer 4: 模型代理层 ──
    add_layer_box(root, parent, 20, 430, 770, 90, "🔀 模型代理层 (Anthropic API Proxy → 172.16.105.104:3000)", "layer_box_cyan")
    add_component(root, parent, 35, 465, 230, 40, "deepseek-v4-pro [1M] — Opus 级", "component_blue")
    add_component(root, parent, 280, 465, 230, 40, "qwen3.7-max [1M] — Sonnet 级", "component_blue")
    add_component(root, parent, 525, 465, 250, 40, "glm-5.1 — Haiku 级", "component_blue")

    # ── Layer 5: Multica 平台 ──
    add_layer_box(root, parent, 20, 540, 770, 350, "☁️ Multica 平台", "layer_box")

    # Workspace / Project / Squad
    add_component(root, parent, 35, 575, 230, 35, "Workspace: vastdata-ai", "component")
    add_component(root, parent, 280, 575, 230, 35, "Project: Vastbase 生态适配", "component")
    add_component(root, parent, 525, 575, 250, 35, "Squad: eco-adapter-team", "component")

    # 10 Agents sub-box
    add_vertex(root, parent, 35, 625, 355, 250, STYLES["layer_box"], "🤖 10 Agents")
    agents = [
        ("task-dispatcher", "08c6ff9d"), ("eco-issue-analyst", "b6d1747f"),
        ("convention-extractor ✨v5", "e9fd48d4"), ("test-scout", "c3ae862b"),
        ("eco-issue-splitter", "3b92a095"), ("adapter-dev", "0544999e"),
        ("code-reviewer", "cdf47ee6"), ("test-adapter", "ed54586d"),
        ("bug-fixer", "a0f924f2"), ("board-watchdog", "ab7e6e9a"),
    ]
    for i, (name, aid) in enumerate(agents):
        row = i // 2
        col = i % 2
        add_component(root, parent, 45 + col * 175, 658 + row * 40, 168, 32, f"{name}", "agent_box")

    # Platform Skills sub-box
    add_vertex(root, parent, 410, 625, 365, 250, STYLES["layer_box_green"], "📦 Platform Skills")
    plat_skills = [
        "vastbase-python-sdk", "multica-cli-reference",
        "adapter-dev-vastbase", "code-reviewer-vastbase",
        "test-adapter-vastbase", "gitflow-branching",
    ]
    for i, ps in enumerate(plat_skills):
        row = i // 2
        col = i % 2
        add_component(root, parent, 420 + col * 175, 658 + row * 50, 168, 32, ps, "skill_box")

    # ── Layer 5b: Git 仓库层 ──
    add_layer_box(root, parent, 810, 60, 350, 350, "📁 Git 仓库层", "layer_box_purple")
    add_component(root, parent, 825, 100, 320, 30, "main (仅 README)", "component")
    add_component(root, parent, 825, 145, 320, 45, "feature/langchain-vastbase-vectorstore\n11 commits", "component_green")
    add_component(root, parent, 825, 200, 320, 45, "feature/llamaindex-vastbase-vector-store\n15 commits", "component_green")
    add_component(root, parent, 825, 260, 320, 55, "agent/test-adapter/* (×5)\nAgent Worktrees（自动清理）", "component_yellow")

    # ── Layer 6: 产出物 ──
    add_layer_box(root, parent, 810, 430, 350, 350, "📄 产出物", "layer_box_green")
    add_component(root, parent, 825, 470, 320, 45, ".multica/conventions/\nlangchain-vectorstore.yaml", "component_green")
    add_component(root, parent, 825, 525, 320, 55, "适配代码包\nlangchain_vastbase / llama_index/...", "component_green")
    add_component(root, parent, 825, 590, 320, 55, "测试套件\nunit + integration + acceptance", "component_green")
    add_component(root, parent, 825, 655, 320, 50, "Multica Issues\nVAS-7 ~ VAS-12+", "component_green")

    # ── Layer 6b: 关键技术栈 (side panel) ──
    add_layer_box(root, parent, 1180, 60, 390, 350, "🔧 关键技术栈", "layer_box_orange")
    add_component(root, parent, 1195, 100, 360, 40, "数据库: Vastbase G100 (PG 线协议兼容)", "component")
    add_component(root, parent, 1195, 155, 360, 40, "SDK: pyvastbase 0.2.6", "component")
    add_component(root, parent, 1195, 210, 360, 55, "LangChain: langchain-core + langchain-postgres\n→ VastbaseVectorStore (1162行)", "component")
    add_component(root, parent, 1195, 280, 360, 55, "LlamaIndex: llama-index-core +\nllama-index-vector-stores-postgres\n→ VastbaseVectorStore (648行)", "component")

    # ── Arrows between layers ──
    # (simplified connectivity — key flows)
    # User → Skills
    # User → Multica
    # Skills → Config
    # Config → Proxy
    # Proxy → Multica
    # Multica → Repo
    # Repo → Artifacts
    # Memory → Skills

    # For brevity, we'll use dashed arrows to indicate data flow directions
    # Between the layer boxes rather than every component

    return root


# ═══════════════════════════════════════════════════════════
# PAGE 2: v5 工作流全景
# ═══════════════════════════════════════════════════════════
def build_page2():
    root = make_root()
    parent = "1"

    add_vertex(root, parent, 500, 5, 400, 35, STYLES["title"], "Multica v5 工作流全景")

    # ── Phase 0: Issue 创建 ──
    y0 = 55
    add_layer_box(root, parent, 20, y0, 480, 200, "📝 Phase 0: Issue 创建", "layer_box_orange")
    steps = [
        "1. 调研上游源码 (PyPI/GitHub)",
        "2. 按模板撰写 Issue",
        "3. 写入本地 .md",
        "4. 🛑 用户审核确认",
        "5. multica issue create",
        "6. 打 '设计' 标签",
    ]
    for i, s in enumerate(steps):
        cid = add_component(root, parent, 35, y0 + 30 + i * 26, 450, 22, s, "component_orange")
        if i > 0:
            add_edge(root, parent, prev, cid, STYLES["arrow"])
        prev = cid

    # ── task-dispatcher ──
    disp_id = add_component(root, parent, 210, 270, 140, 35, "task-dispatcher\n读 label → 路由", "agent_box")
    add_edge(root, parent, prev, disp_id, STYLES["arrow"])

    # ── Phase 1: 父 Issue 分析与设计 ──
    y1 = 330
    add_layer_box(root, parent, 20, y1, 480, 240, "📋 Phase 1: 父 Issue 分析与设计", "layer_box_blue")
    pa_ids = []
    pa_labels = [
        ("eco-issue-analyst", "需求分析 + 方案设计"),
        ("✨ convention-extractor (v5)", "分析上游 → .multica/conventions/*.yaml"),
        ("test-scout 🚫不可跳过", "识别 + 准备框架官方测试"),
        ("eco-issue-splitter", "拆分为子 Issue"),
    ]
    for i, (name, desc) in enumerate(pa_labels):
        aid = add_component(root, parent, 35, y1 + 30 + i * 50, 450, 42, f"{name}\n{desc}", "component_blue")
        if pa_ids:
            add_edge(root, parent, pa_ids[-1], aid, STYLES["arrow"])
        pa_ids.append(aid)
    add_edge(root, parent, disp_id, pa_ids[0], STYLES["arrow"])

    # ── Phase 2: 子 Issue 开发 ──
    y2 = 590
    add_layer_box(root, parent, 20, y2, 480, 280, "🔧 Phase 2: 子 Issue 开发 (并行/串行)", "layer_box_green")

    # TDD Loop
    tdd_y = y2 + 30
    add_vertex(root, parent, 35, tdd_y, 450, 130, STYLES["layer_box_green"], "TDD 开发循环")
    dev_id = add_component(root, parent, 50, tdd_y + 25, 130, 40, "adapter-dev\nStep A4.5 + A6", "agent_box")
    rev_id = add_component(root, parent, 200, tdd_y + 25, 130, 40, "code-reviewer\n6 维审查", "agent_box")
    test_id = add_component(root, parent, 350, tdd_y + 25, 120, 40, "test-adapter\nFlow A", "agent_box")
    add_edge(root, parent, dev_id, rev_id, STYLES["arrow"])
    add_edge(root, parent, rev_id, test_id, STYLES["arrow"], "通过")
    add_edge(root, parent, rev_id, dev_id, STYLES["dashed_arrow"], "不通过→反馈")

    # Gates
    gate_y = tdd_y + 140
    add_vertex(root, parent, 35, gate_y, 450, 110, STYLES["layer_box_red"], "🧪 硬性测试门禁")
    g1_id = add_component(root, parent, 50, gate_y + 25, 195, 30, "🔴 Gate 1: pyvastbase 集成测试", "gate_red")
    g2_id = add_component(root, parent, 260, gate_y + 25, 210, 30, "🔴 Gate 2: 框架官方测试", "gate_red")
    add_edge(root, parent, g1_id, g2_id, STYLES["arrow"])
    block1_id = add_component(root, parent, 60, gate_y + 65, 170, 30, "🚫 阻塞 @luoyj", "block")
    block2_id = add_component(root, parent, 280, gate_y + 65, 170, 30, "🚫 阻塞 @luoyj", "block")
    add_edge(root, parent, g1_id, block1_id, STYLES["arrow"], "失败")
    add_edge(root, parent, g2_id, block2_id, STYLES["arrow"], "失败")
    done_child_id = add_component(root, parent, 150, gate_y + 90, 180, 35, "✅ 子 Issue done", "start_end")
    add_edge(root, parent, test_id, g1_id, STYLES["arrow"])
    add_edge(root, parent, g2_id, done_child_id, STYLES["arrow"], "全部通过")

    # Result arrow
    add_edge(root, parent, pa_ids[-1], dev_id, STYLES["arrow"])

    # ── Decision ──
    dec_id = add_vertex(root, parent, 175, 890, 160, 50, STYLES["decision"], "最后一个子 Issue?")
    add_edge(root, parent, done_child_id, dec_id, STYLES["arrow"])

    # ── Phase 3: Flow C ──
    y3 = 960
    add_layer_box(root, parent, 20, y3, 480, 230, "🏁 Phase 3: Flow C 框架集成验收", "layer_box_red")

    fc_labels = [
        "Phase 9.5: 父 Issue done→todo, unassign→test-adapter",
        "test-adapter 编写 test_framework_integration.py",
        "四层全量回归测试",
    ]
    fc_ids = []
    for i, s in enumerate(fc_labels):
        fid = add_component(root, parent, 35, y3 + 30 + i * 40, 450, 35, s, "component_red")
        if fc_ids:
            add_edge(root, parent, fc_ids[-1], fid, STYLES["arrow"])
        fc_ids.append(fid)
    add_edge(root, parent, dec_id, fc_ids[0], STYLES["arrow"], "是")

    # Four layers detail
    layers_box_y = y3 + 155
    four_y = layers_box_y
    layers = ["1. 单元测试", "2. pyvastbase 集成", "3. 框架官方测试", "4. 🔴 Gate 3: 框架集成验收"]
    prev_l = None
    for i, l in enumerate(layers):
        style = "gate_red" if "Gate 3" in l else "component"
        lid = add_component(root, parent, 50 + i * 105, four_y, 98, 30, l, style)
        if prev_l:
            add_edge(root, parent, prev_l, lid, STYLES["arrow"])
        prev_l = lid
    add_edge(root, parent, fc_ids[-1], lid, STYLES["arrow"])  # from last FC to last layer

    done_id = add_component(root, parent, 170, four_y + 45, 200, 30, "✅ in_review → 用户审核 → done", "start_end")

    # "否" loop back
    add_edge(root, parent, dec_id, dev_id, STYLES["dashed_arrow"], "否 (继续下一个子Issue)")

    # ── Right side: Convention Spec flow ──
    rx = 540
    add_layer_box(root, parent, rx, 55, 420, 220, "✨ Convention Spec 数据流 (v5 新增)", "layer_box_purple")
    up_id = add_component(root, parent, rx + 15, 90, 390, 30, "上游框架源码 (GitHub / PyPI)", "component")
    ext_id = add_component(root, parent, rx + 15, 135, 390, 40, "convention-extractor Agent (e9fd48d4)\n分析代码风格·API模式·测试规范·置信度", "agent_box")
    spec_id = add_component(root, parent, rx + 15, 190, 390, 35, "↓ .multica/conventions/*.yaml", "component_green")
    add_edge(root, parent, up_id, ext_id, STYLES["arrow"])
    add_edge(root, parent, ext_id, spec_id, STYLES["arrow"])

    # Consumers
    cons_y = 195
    add_vertex(root, parent, rx + 15, cons_y + 40, 390, 70, STYLES["layer_box"], "下游消费")
    c1 = add_component(root, parent, rx + 25, cons_y + 55, 175, 40, "adapter-dev\nStep A4.5 + A6", "agent_box")
    c2 = add_component(root, parent, rx + 215, cons_y + 55, 175, 40, "code-reviewer\n第6维: 框架规范符合度", "agent_box")

    # ── Right side: Branch strategy ──
    add_layer_box(root, parent, rx, 300, 420, 180, "📁 分支策略", "layer_box")
    add_component(root, parent, rx + 15, 335, 390, 25, "一个项目 = 一个 feature 分支", "component")
    add_component(root, parent, rx + 15, 370, 390, 25, "所有子 Issue 共享同一分支", "component")
    add_component(root, parent, rx + 15, 405, 390, 25, "Agent 在隔离 worktree 中工作", "component")
    add_component(root, parent, rx + 15, 440, 390, 25, "worktree 无变更时自动清理", "component")

    return root


# ═══════════════════════════════════════════════════════════
# PAGE 3: Agent 协作关系
# ═══════════════════════════════════════════════════════════
def build_page3():
    root = make_root()
    parent = "1"

    add_vertex(root, parent, 450, 5, 500, 35, STYLES["title"], "Agent 协作关系图")

    # ── Orchestration Layer ──
    orch_y = 55
    add_layer_box(root, parent, 40, orch_y, 350, 170, "🎯 编排层", "layer_box_orange")
    disp = add_component(root, parent, 55, orch_y + 30, 140, 60, "task-dispatcher\n08c6ff9d\n读 Label → 路由", "agent_box")
    watch = add_component(root, parent, 210, orch_y + 30, 165, 60, "board-watchdog\nab7e6e9a\n每10min 巡逻 Board", "agent_box")
    bug = add_component(root, parent, 130, orch_y + 105, 170, 45, "bug-fixer\na0f924f2\nBug 分析修复", "agent_box")

    # ── Design Layer ──
    des_y = 260
    add_layer_box(root, parent, 40, des_y, 500, 210, "📐 设计层", "layer_box_blue")
    d_analyst = add_component(root, parent, 55, des_y + 30, 150, 60, "eco-issue-analyst\nb6d1747f\n需求分析+方案设计", "agent_box")
    d_conv = add_component(root, parent, 220, des_y + 30, 150, 60, "convention-extractor\ne9fd48d4 ✨v5\n提取上游编码规范", "agent_box")
    d_scout = add_component(root, parent, 385, des_y + 30, 140, 60, "test-scout\nc3ae862b\n框架官方测试准备", "agent_box")
    d_split = add_component(root, parent, 55, des_y + 110, 470, 35, "eco-issue-splitter (3b92a095) — 拆分为子 Issue", "agent_box")

    # design flow arrows
    add_edge(root, parent, d_analyst, d_conv, STYLES["arrow"])
    add_edge(root, parent, d_conv, d_scout, STYLES["arrow"])
    add_edge(root, parent, d_scout, d_split, STYLES["arrow"])

    # ── Development Layer ──
    dev_y = 500
    add_layer_box(root, parent, 40, dev_y, 500, 200, "💻 开发层", "layer_box_green")
    dev_agent = add_component(root, parent, 55, dev_y + 30, 140, 65, "adapter-dev\n0544999e\nTDD 开发\n消费 Convention Spec", "agent_box")
    reviewer = add_component(root, parent, 220, dev_y + 30, 140, 65, "code-reviewer\ncdf47ee6\n6 维审查\n含框架规范符合度", "agent_box")
    tester = add_component(root, parent, 385, dev_y + 30, 140, 65, "test-adapter\ned54586d\n3 种测试流程\nA / B / C", "agent_box")

    add_edge(root, parent, dev_agent, reviewer, STYLES["arrow"], "提交 Review")
    add_edge(root, parent, reviewer, tester, STYLES["arrow"], "通过")
    add_edge(root, parent, reviewer, dev_agent, STYLES["dashed_arrow"], "不通过→反馈")
    add_edge(root, parent, tester, bug, STYLES["dashed_arrow"], "Bug 发现")
    add_edge(root, parent, bug, dev_agent, STYLES["dashed_arrow"], "修复后")

    # ── Dispatcher routing ──
    add_edge(root, parent, disp, d_analyst, STYLES["arrow"], "'设计' label")
    add_edge(root, parent, disp, bug, STYLES["arrow"], "Bug label")
    add_edge(root, parent, disp, dev_agent, STYLES["arrow"], "开发 label")
    add_edge(root, parent, watch, disp, STYLES["dashed_arrow"], "监控")

    # Splitter → dev
    add_edge(root, parent, d_split, dev_agent, STYLES["arrow"])

    # ── Platform Skills (right panel) ──
    rx = 580
    add_layer_box(root, parent, rx, 55, 380, 280, "📦 Platform Skills（注入到 Agent）", "layer_box")
    ps_list = [
        ("adapter-dev-vastbase", "017e93a2", "→ adapter-dev"),
        ("code-reviewer-vastbase", "1f059118", "→ code-reviewer"),
        ("test-adapter-vastbase", "f5dc63b9", "→ test-adapter"),
        ("vastbase-python-sdk", "11f90ebb", "→ adapter-dev, test-adapter"),
        ("multica-cli-reference", "f7ee8612", "→ 所有 Agent"),
        ("gitflow-branching", "cdd82b08", "→ 所有 Agent"),
    ]
    for i, (name, pid, target) in enumerate(ps_list):
        add_component(root, parent, rx + 15, 95 + i * 37, 350, 32, f"{name} ({pid}) {target}", "skill_box")

    # ── Convention Spec artifact ──
    add_layer_box(root, parent, rx, 360, 380, 130, "📄 Convention Spec 流向", "layer_box_purple")
    art = add_component(root, parent, rx + 15, 400, 350, 35, ".multica/conventions/*.yaml", "component_green")
    cons_dev = add_component(root, parent, rx + 15, 445, 170, 35, "→ adapter-dev (Step A4.5+A6)", "agent_box")
    cons_rev = add_component(root, parent, rx + 195, 445, 175, 35, "→ code-reviewer (第6维)", "agent_box")

    # Convention extractor → spec
    add_edge(root, parent, d_conv, art, STYLES["dashed_arrow"], "产出")

    # ── Legend ──
    add_layer_box(root, parent, rx, 520, 380, 150, "📋 图例", "layer_box")
    add_component(root, parent, rx + 15, 555, 80, 25, "实线箭头 = 主流程", "component")
    add_component(root, parent, rx + 110, 555, 120, 25, "虚线箭头 = 反馈/监控/注入", "component")
    add_component(root, parent, rx + 15, 590, 170, 25, "蓝色 = Agent  |  黄色 = Skill", "component")
    add_component(root, parent, rx + 195, 590, 170, 25, "紫色 = Artifact  |  红色 = 门禁", "component")

    return root


# ═══════════════════════════════════════════════════════════
# PAGE 4: 测试门禁体系
# ═══════════════════════════════════════════════════════════
def build_page4():
    root = make_root()
    parent = "1"

    add_vertex(root, parent, 450, 10, 450, 35, STYLES["title"], "测试门禁体系 — 三道硬性门禁")

    # ── Child Issue Test Flow ──
    cy = 60
    add_layer_box(root, parent, 20, cy, 550, 350, "子 Issue 测试流程", "layer_box")
    ct1 = add_component(root, parent, 200, cy + 30, 180, 35, "Phase 2a: 单元测试 (pytest)", "component")
    ct2 = add_component(root, parent, 180, cy + 85, 220, 40, "🔴 Gate 1: pyvastbase 集成测试\n连接真实 Vastbase", "gate_red")
    ct3 = add_component(root, parent, 180, cy + 145, 220, 40, "🔴 Gate 2: 框架官方测试\n上游框架自带测试", "gate_red")
    ct4 = add_component(root, parent, 210, cy + 205, 160, 35, "✅ 子 Issue 通过", "start_end")
    block1 = add_component(root, parent, 50, cy + 250, 180, 30, "🚫 阻塞 (不可标 known bug)", "block")
    block2 = add_component(root, parent, 350, cy + 250, 180, 30, "🚫 阻塞 @luoyj", "block")

    add_edge(root, parent, ct1, ct2, STYLES["arrow"])
    add_edge(root, parent, ct2, ct3, STYLES["arrow"], "通过")
    add_edge(root, parent, ct2, block1, STYLES["arrow"], "失败")
    add_edge(root, parent, ct3, ct4, STYLES["arrow"], "通过")
    add_edge(root, parent, ct3, block2, STYLES["arrow"], "缺失/失败")

    # ── Parent Issue Flow C ──
    py = 430
    add_layer_box(root, parent, 20, py, 550, 380, "父 Issue Flow C: 框架集成验收", "layer_box_red")
    pt1 = add_component(root, parent, 170, py + 30, 240, 35, "编写 test_framework_integration.py\n使用框架高层 API", "component_red")
    pt2 = add_component(root, parent, 150, py + 80, 280, 40, "VectorStoreIndex.from_documents()\nas_query_engine() 完整 RAG 链路", "component_red")
    pt3 = add_component(root, parent, 200, py + 135, 180, 35, "四层全量回归 ↓", "component_red")

    layers = ["L1: 单元测试", "L2: pyvastbase 集成", "L3: 框架官方测试", "L4: 🔴 Gate 3 框架集成验收"]
    prev_l = pt3
    for i, l in enumerate(layers):
        style = "gate_red" if "Gate 3" in l else "component"
        lid = add_component(root, parent, 40 + i * 125, py + 185, 118, 35, l, style)
        if i > 0:
            add_edge(root, parent, prev_lid, lid, STYLES["arrow"])
        prev_lid = lid

    pt_final = add_component(root, parent, 180, py + 240, 220, 35, "✅ 全部通过 → in_review → done", "start_end")
    block3 = add_component(root, parent, 60, py + 295, 200, 30, "🚫 阻塞，不可降级", "block")
    add_edge(root, parent, prev_lid, pt_final, STYLES["arrow"], "全部通过")
    add_edge(root, parent, prev_lid, block3, STYLES["arrow"], "失败")

    # ── Flow between child and parent ──
    add_edge(root, parent, ct4, pt1, STYLES["arrow"], "所有子 Issue done")

    # ── Right: Gate Summary Table ──
    rx = 610
    add_layer_box(root, parent, rx, 60, 500, 250, "📋 三道门禁总览", "layer_box")

    # Table-like layout
    headers = ["Gate", "内容", "触发时机", "失败处理"]
    col_x = [rx + 15, rx + 80, rx + 280, rx + 390]
    for j, h in enumerate(headers):
        add_component(root, parent, col_x[j], 95, 95 if j == 1 else 90, 25, h, "component")

    rows = [
        ("Gate 1", "pyvastbase 集成测试", "每子 Issue Phase 2b", "🚫 阻塞，不可标 known bug"),
        ("Gate 2", "框架官方测试", "每子 Issue Phase 3", "🚫 阻塞 @luoyj"),
        ("Gate 3", "框架集成验收", "父 Issue Flow C", "🚫 阻塞，不可降级"),
    ]
    for i, (gate, content, when, fail) in enumerate(rows):
        ry = 130 + i * 35
        style_g = "gate_red" if i == 2 else "component_red"
        add_component(root, parent, col_x[0], ry, 55, 28, gate, style_g)
        add_component(root, parent, col_x[1], ry, 190, 28, content, "component")
        add_component(root, parent, col_x[2], ry, 100, 28, when, "component")
        add_component(root, parent, col_x[3], ry, 100, 28, fail, "component_red")

    # ── Right: VAS-7 Lessons ──
    add_layer_box(root, parent, rx, 340, 500, 200, "📖 VAS-7 教训 (推动三道门禁的建立)", "layer_box_orange")
    lessons = [
        "• pyvastbase 集成测试发现 9 个兼容性问题 (单元测试遗漏)",
        "• 零框架级测试 → 适配代码从未走完整 RAG 链路验证",
        "• 子 Issue 各自通过 ≠ 组装后没问题 (JSONB filter bug)",
        "• test-scout 全程未交付框架官方测试",
        "• test-adapter 两次试图跳过集成测试",
    ]
    for i, l in enumerate(lessons):
        add_vertex(root, parent, rx + 15, 378 + i * 28, 470, 24, STYLES["label"], l)

    return root


# ═══════════════════════════════════════════════════════════
# PAGE 5: 适配项目产出全景
# ═══════════════════════════════════════════════════════════
def build_page5():
    root = make_root()
    parent = "1"

    add_vertex(root, parent, 400, 5, 500, 35, STYLES["title"], "适配项目产出全景 & 技术栈")

    # ── LangChain Adapter ──
    add_layer_box(root, parent, 20, 55, 400, 330, "🔗 LangChain 适配 (feature/langchain-vastbase-vectorstore)", "layer_box_blue")

    lc_files = [
        ("__init__.py", "导出: VastbaseVectorStore, Filter, utils"),
        ("vectorstore.py (1162行)", "sync/async CRUD + dense/hybrid 搜索"),
        ("utils.py (211行)", "MetadataFilters → SQL WHERE 翻译"),
        ("pyproject.toml", "langchain-vastbase v0.1.0 | Python>=3.9"),
    ]
    for i, (f, desc) in enumerate(lc_files):
        add_component(root, parent, 35, 93 + i * 55, 370, 48, f"{f}\n{desc}", "component_blue")

    # Tests
    add_vertex(root, parent, 35, 325, 370, 50, STYLES["layer_box_green"], "Tests")
    add_component(root, parent, 45, 345, 110, 25, "test_vectorstore.py (1069行)", "component_green")
    add_component(root, parent, 165, 345, 110, 25, "test_utils.py (298行)", "component_green")
    add_component(root, parent, 285, 345, 110, 25, "conftest.py (67行)", "component_green")

    # ── LlamaIndex Adapter ──
    add_layer_box(root, parent, 440, 55, 420, 330, "🦙 LlamaIndex 适配 (feature/llamaindex-vastbase-vector-store)", "layer_box_green")

    li_files = [
        ("__init__.py", "导出: VastbaseVectorStore, _to_vastbase_filter"),
        ("base.py (648行)", "CRUD + DENSE/HYBRID/TEXT search"),
        ("utils.py (153行)", "MetadataFilters → SQL WHERE 翻译"),
        ("pyproject.toml", "llama-index-vector-stores-vastbase v0.1.0 | Python>=3.11"),
    ]
    for i, (f, desc) in enumerate(li_files):
        add_component(root, parent, 455, 93 + i * 55, 390, 48, f"{f}\n{desc}", "component_green")

    # Tests
    add_vertex(root, parent, 455, 325, 390, 50, STYLES["layer_box_green"], "Tests (119 tests total)")
    add_component(root, parent, 465, 345, 85, 25, "test_vs.py (51t)", "component_green")
    add_component(root, parent, 558, 345, 95, 25, "test_filter.py (35t)", "component_green")
    add_component(root, parent, 660, 345, 85, 25, "test_int.py (23t)", "component_green")
    add_component(root, parent, 753, 345, 85, 25, "test_supp.py (10t)", "component_green")

    # ── Search Capabilities ──
    add_layer_box(root, parent, 20, 405, 840, 130, "🔍 搜索能力矩阵", "layer_box")
    cap_labels = [
        ("DENSE\n向量相似度", "COSINE / L2 / IP\nVastbaseCollection.ann_search()"),
        ("HYBRID\n混合搜索", "BM25 全文 + DENSE 向量\nReciprocal Rank Fusion (RRF)"),
        ("TEXT_SEARCH\n全文检索", "BM25 (pyvastbase)\nILIKE fallback (LlamaIndex)"),
    ]
    for i, (title, desc) in enumerate(cap_labels):
        add_component(root, parent, 35 + i * 270, 445, 255, 75, f"{title}\n{desc}", "component_blue")

    # ── Convention Spec ──
    add_layer_box(root, parent, 20, 555, 400, 180, "📋 Convention Spec (.multica/conventions/)", "layer_box_purple")
    spec_sections = [
        "project_structure — 包命名 / 模块布局 / pyproject 要求",
        "code_style — import 顺序 / 命名 / 类型注解 / docstring",
        "api_patterns — 工厂方法 / 错误处理 / 连接管理 / 异步策略",
        "serialization — JSON 处理 / 列类型 (UUID PK, JSONB, VECTOR)",
        "test_conventions — pytest / fixture 模式 / 命名规范",
    ]
    for i, s in enumerate(spec_sections):
        add_component(root, parent, 35, 593 + i * 26, 370, 22, s, "component")

    # ── Infrastructure ──
    add_layer_box(root, parent, 440, 555, 420, 180, "🏗️ 基础设施依赖", "layer_box_orange")
    infra = [
        ("Claude Code CLI", "AI 编程助手 · superpowers plugin v5.1.0"),
        ("Multica 平台", "Agent 编排 · Issue 追踪 · Squad 协作"),
        ("Git Worktrees", "Agent 隔离工作空间 · 自动清理"),
        ("pyvastbase 0.2.6", "Vastbase Python SDK · VastbaseClient · AsyncCollection"),
        ("pytest", "测试框架 · 119+ tests · 3 层门禁"),
    ]
    for i, (name, desc) in enumerate(infra):
        add_component(root, parent, 455, 593 + i * 26, 390, 22, f"{name}: {desc}", "component_orange")

    return root


# ═══════════════════════════════════════════════════════════
# BUILD MXFILE
# ═══════════════════════════════════════════════════════════
def build_mxfile():
    mxfile = ET.Element(
        "mxfile",
        host="app.diagrams.net",
        modified=datetime.datetime.now().isoformat(),
        agent="Multica Architecture Generator",
        version="21.0.0",
        type="device",
    )

    pages = [
        ("🏠 系统全景图", build_page1),
        ("🔄 v5 工作流", build_page2),
        ("🤖 Agent 协作关系", build_page3),
        ("🧪 测试门禁体系", build_page4),
        ("📦 项目产出 & 技术栈", build_page5),
    ]

    for name, builder in pages:
        CELL_ID[0] = 2  # reset counter per page
        root = builder()
        diagram = build_diagram(name, root, 1600, 1200)
        mxfile.append(diagram)

    # Pretty-print
    rough = ET.tostring(mxfile, "utf-8")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ", encoding="utf-8")


if __name__ == "__main__":
    xml_str = build_mxfile()
    out_path = "/Users/Admin/Desktop/生态适配/vastdata-ai/docs/multica-architecture.drawio"
    with open(out_path, "wb") as f:
        f.write(xml_str)
    print(f"✅ DrawIO file generated: {out_path}")
    print(f"   Contains pages: 系统全景图, v5工作流, Agent协作关系, 测试门禁体系, 项目产出&技术栈")
