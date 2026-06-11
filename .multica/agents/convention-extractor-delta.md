## Framework Profile 读取（v6 新增）

在提取编码规范之前，**必须先读取** Parent Issue 评论区中的 Framework Profile JSON。

### 提取来源分流

根据 `integration_mode` 切换规范提取的参照对象：

#### standalone 模式（独立包）

维持现有逻辑：
- 分析框架 VectorStore 接口定义文件
- 提取命名规范、类型注解风格、文档字符串格式

#### native 模式（原生集成）

切换到以参照后端为主的提取策略：
- **主要参照**：`convention_references` 中列出的现有后端实现文件
- **次要参照**：项目根目录的编码风格（通过扫描周边 Python 文件的 import 顺序、类型注解、注释风格等）
- 提取内容包括：
  - 类命名规范（如 `InfinityDB` → 新后端命名模式）
  - 方法命名和参数风格（参数名、默认值模式）
  - 配置项命名规范
  - 日志和错误处理模式
  - import 组织方式（相对导入 vs 绝对导入）
  - 注册代码的写法模式（如何在 factory/__init__ 中注册）

#### plugin 模式

分析插件规范文件中的编码约定（如 setup.py entry_point 格式、插件元数据格式）。
