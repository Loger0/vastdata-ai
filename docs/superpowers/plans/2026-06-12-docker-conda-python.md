# Docker Conda Python 3.11 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个基于 `continuumio/miniconda3` 的 Docker 镜像，内含 Python 3.11 的 conda 环境，供手动 `docker exec` 调用。

**Architecture:** 单一 Dockerfile，使用 multi-line RUN 指令创建 conda 环境并验证。容器通过 `tail -f /dev/null` 常驻后台，用户通过 `docker exec ... conda run -n py311 python` 执行脚本。

**Tech Stack:** Docker, continuumio/miniconda3, Conda, Python 3.11

---

### Task 1: 编写 Dockerfile

**Files:**
- Create: `docker/Dockerfile.conda-py311`

- [ ] **Step 1: 创建 Dockerfile**

```dockerfile
FROM continuumio/miniconda3:latest

# 创建 Python 3.11 的 conda 环境
RUN conda create -n py311 python=3.11 -y \
    && conda clean -afy

# 验证安装
RUN conda run -n py311 python --version \
    && conda run -n py311 pip --version

# 设置默认启动命令（保持容器运行）
CMD ["tail", "-f", "/dev/null"]
```

### Task 2: 构建镜像并验证

- [ ] **Step 1: 构建镜像**

```bash
docker build -f docker/Dockerfile.conda-py311 -t conda-py311 .
```

Expected: 构建成功，无错误输出。

- [ ] **Step 2: 验证 Python 版本**

```bash
docker run --rm conda-py311 conda run -n py311 python --version
```

Expected: `Python 3.11.x`

- [ ] **Step 3: 验证可执行路径指向 conda 环境内**

```bash
docker run --rm conda-py311 conda run -n py311 python -c "import sys; print(sys.executable)"
```

Expected: 输出路径包含 `/opt/conda/envs/py311`

- [ ] **Step 4: 验证 pip 可用**

```bash
docker run --rm conda-py311 conda run -n py311 pip list
```

Expected: 列出环境中已安装的包（pip, setuptools, wheel 等）。

### Task 3: 提交

- [ ] **Step 1: 提交代码**

```bash
git add docker/Dockerfile.conda-py311
git commit -m "feat: add Docker Conda Python 3.11 environment"
```
