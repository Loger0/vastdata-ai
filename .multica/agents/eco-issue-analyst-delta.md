## Framework Profile 读取（v6 新增）

在开始需求分析之前，**必须先读取** Parent Issue 评论区中的 Framework Profile JSON。

### 读取步骤

1. 获取 Parent Issue 的评论列表
2. 找到由 Framework Analyzer 发布的最新评论，提取其中的 JSON block
3. 解析 JSON，获取以下关键字段：
   - `integration_mode`: 集成模式
   - `coupling_level`: 耦合度
   - `files_to_modify`: 需修改的文件清单
   - `files_to_create`: 需新增的文件清单
   - `base_class`: 抽象基类
   - `required_methods`: 必须实现的方法列表
   - `test_strategy`: 测试策略
   - `demo`: Demo 规划

### 模式分流

根据 `integration_mode` 切换需求分析策略：

#### standalone 模式（独立包）

维持现有分析逻辑：
- 分析框架的 VectorStore 接口规范
- 确定适配器需要实现的方法和参数签名
- 分析元数据过滤、混合搜索等扩展能力

#### native 模式（原生集成）

切换到原生集成分析策略：
- 分析 `files_to_modify` 中每个文件的现有逻辑，确定修改范围
- 分析 `reference_backend` 的实现，理解后端注册和调用机制
- 确定新增文件需要实现的全部方法
- 分析配置项、枚举值、工厂注册等分发逻辑的改动点
- 特别注意：修改目标框架文件时不能破坏现有后端的行为

#### plugin 模式（插件机制）

分析框架的插件注册规范：
- 确定插件入口文件格式
- 分析插件发现和加载机制
- 确定配置方式（setup.py entry_point、配置文件声明等）

### Profile 缺失处理

如果 Issue 评论区没有 Framework Profile JSON，这是异常状态。暂停分析并在 Issue 评论区发出警告：
"⚠️ 未找到 Framework Profile。请先将此 Issue 重新指派给 Framework Analyzer 完成框架诊断。"
