## Framework Profile 读取（v6 新增）

在执行测试之前，**必须先读取** Parent Issue 评论区中的 Framework Profile JSON。

### 测试策略分流

根据 `test_strategy` 切换测试执行策略：

#### test_strategy = "extract_official"（框架有官方测试）

执行 4 层测试（维持现有流程 C）：
1. 单元测试
2. pyvastbase 集成测试
3. 框架官方测试
4. 框架集成验收

#### test_strategy = "extend_partial"（部分覆盖）

执行 4 层 + 补充自建测试：
1. 单元测试
2. pyvastbase 集成测试
3. 框架官方测试（已有的）
4. 补充测试（按 test-strategist 的 TEST_PLAN 补充缺失场景）
5. 框架集成验收

#### test_strategy = "self_build_e2e"（完全缺失，核心变化）

执行全自建测试流程：

##### 子 Issue 级别测试（Phase 2b-3）

读取 test-strategist 产出的 `TEST_PLAN.md`，针对本子 Issue 负责的 `required_methods` 编写测试：

- **单元测试**：对纯逻辑部分（如 filter 转换、SQL 生成）编写单元测试
- **集成测试**：对每个方法编写真实 Vastbase 连接下的测试
  - 正常路径：基础功能的正确性
  - 边界条件：空输入、极限值
  - 错误处理：无效参数、不存在资源

##### 父 Issue Flow C（升级为 5 层）

所有子 Issue done 后，执行以下 5 层验收：

| 层 | 名称 | 执行内容 |
|:---:|------|------|
| 1 | 单元测试 | 汇总所有子 Issue 的单元测试 |
| 2 | pyvastbase 集成 | 汇总所有子 Issue 的 pyvastbase 集成测试 |
| 3 | 自建集成测试 | 运行 TEST_PLAN 中所有跨方法的集成测试用例 |
| 4 | 框架集成验收 | 使用框架高层 API 验证完整适配链路 |
| 5 🆕 | **应用级 Demo** | 基于 Profile 的 `demo.scenarios` 编写并运行 Demo |

### 第 5 层：应用级 Demo 编写规范

#### Demo 脚本位置

```
tests/demo/test_framework_integration_demo.py
```

#### Demo 编写要求

1. **必须覆盖 `demo.scenarios` 中的所有场景**
2. **必须是自包含脚本**：一个 `python tests/demo/test_framework_integration_demo.py` 跑通全部流程
3. **必须输出明确的 PASS/FAIL**：每个 scenario 结束时打印 `[PASS]` 或 `[FAIL]`
4. **必须包含数据清理**：Demo 结束后清理测试数据，不留残留
5. **必须是 API 层面的**：不操作 Web UI，纯 Python API 调用

#### Demo 脚本模板（以 native 模式为例）

```python
"""RAGFlow Vastbase 后端 — 应用级集成 Demo

Profile 要求覆盖的场景：
- knowledge_base_create
- document_ingest_and_parse
- vector_retrieval
- hybrid_search_recall
- knowledge_base_cleanup
"""

import sys
from pyvastbase import connect

# Vastbase 连接配置
VASTBASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vastbase",
    "user": "admin",
    "password": "password"
}

def test_scenario(name):
    """装饰器风格的场景标记"""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"Scenario: {name}")
            print(f"{'='*60}")
            try:
                func()
                print(f"[PASS] {name}")
                return True
            except Exception as e:
                print(f"[FAIL] {name}: {e}")
                return False
        return wrapper
    return decorator

@test_scenario("knowledge_base_create")
def test_kb_create():
    """验证：RAGFlow 知识库创建 + Vastbase Collection 映射"""
    # 调用 RAGFlow API 创建 Knowledge Base
    # 验证 Vastbase 中对应 Collection 被创建
    pass

@test_scenario("document_ingest_and_parse")
def test_doc_ingest():
    """验证：文档上传 + 解析 + 向量化 + 入库"""
    # 上传文档到 RAGFlow KB
    # 等待解析完成
    # 验证 Vastbase 中有对应的向量记录
    pass

# ... 其他 scenarios

def main():
    results = []
    results.append(test_kb_create())
    results.append(test_doc_ingest())
    # ... 运行所有 scenarios

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Demo 结果: {passed}/{total} 通过")
    print(f"{'='*60}")

    if passed == total:
        print("✅ 全部场景通过！")
        sys.exit(0)
    else:
        print(f"❌ {total - passed} 个场景失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 硬性约束

- **Demo 失败 = 阻塞**：不可降级为"已知问题"，不可跳过
- **所有 scenarios 必须通过**：任何一个 scenario 失败整体验收不通过
- **必须清理数据**：Demo 结束后不留测试残留
