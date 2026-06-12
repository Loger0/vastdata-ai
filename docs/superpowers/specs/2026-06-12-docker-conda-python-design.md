# Docker Conda Python 3.11 运行环境

**日期**: 2026-06-12
**状态**: 设计中
**背景**: macOS 本机存在多个 Python 版本（python@3.11, python@3.14, miniforge 3.13, pyenv），Claude Code Agent 运行时执行 Python 命令时容易落入错误环境，导致循环错误。

## 目标

提供一个基于 Docker + Conda 的独立 Python 3.11 运行环境，由用户手动通过 `docker exec` 调用，彻底隔离本机 Python 版本混乱问题。

## 设计

### 单一交付物：Dockerfile

一个 Dockerfile，内容如下：

- **Base 镜像**: `continuumio/miniconda3:latest`（官方 miniconda 镜像，内置 conda）
- **Conda 环境**: 创建名为 `py311` 的环境，指定 Python 3.11
- **基础包**: pip、setuptools、wheel
- **验证**: `conda run -n py311 python --version` 输出 3.11.x

### 使用流程

```bash
# 1. 构建镜像
docker build -t conda-py311 .

# 2. 启动容器（常驻后台，挂载工作目录）
docker run -d --name py311 \
  -v /path/to/your/work:/work \
  conda-py311 tail -f /dev/null

# 3. 执行 Python
docker exec py311 conda run -n py311 python /work/script.py

# 4. 进入容器交互
docker exec -it py311 conda run -n py311 bash
```

### 文件位置

```
vastdata-ai/
  docker/
    Dockerfile.conda-py311
```

## 非目标

- 不自动拦截/代理本机 `python` 命令
- 不常驻后台（用户自行管理容器生命周期）
- 不预装数据科学或 ML 包（按需后续添加）

## 验证标准

- `docker build` 成功
- `docker exec ... python --version` 输出 `Python 3.11.x`
- `docker exec ... python -c "import sys; print(sys.executable)"` 指向 conda 环境内路径
