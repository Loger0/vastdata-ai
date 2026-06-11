## Framework Profile 读取（v6 新增）

在执行测试侦察之前，**必须先读取** Parent Issue 评论区中的 Framework Profile JSON。

### 测试基础分流

根据 Profile 中的 `test_infrastructure` 字段切换行为：

#### test_infrastructure = "full"（框架有官方测试）

维持现有 test-scout 行为：
- 提取框架官方 VectorStore 测试套件
- 将测试用例列表交付给 test-adapter
- **硬性门禁**：框架官方测试必须交付，不可自行决定跳过

#### test_infrastructure = "partial"（部分覆盖）

部分扩展模式：
- 提取已有的官方测试
- 标注缺失的测试覆盖（对比 `required_methods` 清单）
- 对缺失部分规划补充测试用例

#### test_infrastructure = "none"（完全缺失）→ 切换为 test-strategist 角色

**角色：test-strategist（测试策略规划者）**

当目标框架没有任何可参考的测试用例时，你的职责是规划一份完整的集成测试方案。

##### 规划输入

- Profile 中的 `required_methods`：必须测试的方法列表
- Profile 中的 `demo.scenarios`：应用级验证场景
- Profile 中的 `base_class` / `reference_backend`：接口规范和参照实现
- Profile 中的 `integration_mode`：测试的代码位置

##### 规划输出

产出 `TEST_PLAN.md`，包含：

1. **测试范围**：列出所有需测试的方法和场景
2. **测试用例清单**：每个方法对应 2-4 个测试用例
   - 正常路径（happy path）
   - 边界条件（空输入、极限值）
   - 错误处理（无效参数、连接失败）
3. **集成测试用例**：跨方法的流程测试（增→查→改→删）
4. **Demo 测试脚本**：基于 `demo.scenarios` 的端到端验证脚本规划
5. **测试数据**：需要的测试文档/向量/过滤条件
6. **预期结果**：每个用例的期望输出

##### 测试用例命名规范

```
test_<method>_<scenario>_<expected_behavior>
```

示例：
- `test_search_dense_vector_returns_topk_results`
- `test_insert_duplicate_id_raises_error`
- `test_delete_nonexistent_collection_handles_gracefully`

##### 交付方式

将 `TEST_PLAN.md` 作为附件发布到 Issue 评论区，供 test-adapter 消费。

## 硬性约束

- **不可跳过**：即使是 `none` 也必须产出完整的 TEST_PLAN，不可说"没有测试所以不写"
- **不可模糊**：每个测试用例必须有明确的输入、操作、预期输出
- **必须基于 Profile**：所有方法名和场景必须与 Profile 中的 `required_methods` 和 `demo.scenarios` 对应
