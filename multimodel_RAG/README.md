# 多模态 Agentic RAG 旅游咨询智能系统

基于 **LangGraph** 编排的中心化监管架构，融合 **Naive RAG**、**GraphRAG** 和 **多模态 VLM RAG** 三大智能体，实现中国大陆及港澳台旅游咨询的全场景覆盖。

## 项目架构

```
用户提问
    │
    ▼
┌─────────────────┐
│  中心路由智能体   │  ← LLM 结构化输出，判断问题归属
│ (rag_router)     │
└────┬────────────┘
     │ 条件路由
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  ▼
┌──────────┐    ┌──────────────┐    ┌─────────────────┐
│ Naive RAG │    │  GraphRAG    │    │ Multi-Model RAG │
│  Agent    │    │   Agent      │    │    Agent        │
│(中国大陆) │    │ (中国大陆)    │    │   (港澳台)       │
│ 基础咨询  │    │ 高级分析咨询  │    │  多模态文档检索  │
└──────────┘    └──────────────┘    └─────────────────┘
```

### 三大智能体分工

| 智能体 | 适用场景 | 检索方式 | 数据来源 |
|--------|---------|---------|---------|
| **rag_agent** | 中国大陆基础旅游咨询（景点介绍、交通线路、住宿餐饮等） | FAISS 向量检索 | travel_guide.csv 结构化数据 |
| **graphrag_agent** | 中国大陆高级旅游分析（多景点比较、主题分析、概念关联等） | GraphRAG 知识图谱全局检索 | 旅游攻略文本知识图谱 |
| **multi_model_agent** | 港澳台旅游咨询（攻略、行程、签证、交通等） | FAISS 向量检索 | VLM/OCR 解析的 PDF 攻略文档 |

## 核心特性

### 1. 智能路由分发
中心路由智能体使用 DeepSeek LLM + 结构化输出（`with_structured_output`），结合历史对话上下文，精准判断用户问题应由哪个智能体处理。

### 2. 自纠错 RAG 流程
每个 RAG 智能体均内置 **检索-评分-改写** 循环：
- **检索**：调用对应工具进行知识检索
- **评分**：LLM 评估检索结果与用户问题的相关性
- **改写**：若不相关，自动改写用户问题并重新检索（最多重试 2 次）
- **生成**：基于相关上下文生成最终回答

### 3. GraphRAG 知识图谱
利用 GraphRAG 2.7.0 从旅游攻略文本中构建实体-关系知识图谱，支持：
- **Global Search**：跨社区全局检索，适合概括性、比较性问题
- **Local Search**：局部实体检索，适合具体景点查询
- **Drift Search**：结合全局与局部的混合检索

### 4. 多模态文档处理
支持两种 PDF 文档解析管线，将旅游攻略 PDF 转换为可检索的向量知识库：

#### 管线 A：VLM 视觉语言模型（olmOCR-7B）
```
PDF → pdf2image 逐页转图片 → vLLM 部署 olmOCR-7B → 逐页 OCR 转 Markdown → 向量入库
```
- 支持表格、流程图、路线图的智能识别与摘要
- 适合复杂排版的旅游攻略文档

#### 管线 B：纯 OCR（Tesseract + Unstructured）
```
PDF → Unstructured(hi_res策略) + Tesseract OCR → YOLO 版面检测 → 表格识别 → Markdown → 向量入库
```
- 使用 YOLO 目标检测进行版面分析
- 使用 Table Transformer 进行表格结构识别
- 提供元素检测可视化功能

## 项目结构

```
multimodel_RAG/
├── System_Agent.py                 # 主入口：中心化监管架构 LangGraph 工作流
├── model_init_parameters.json      # YOLO 版面检测模型配置
├── requirements.txt                # Python 依赖
├── .env                            # 环境变量配置
│
├── Naive_RAG/                      # Naive RAG 智能体模块
│   ├── rag_agent.py                #   RAG Agent（检索-评分-改写-生成）
│   └── create_vectorstore.py       #   FAISS 向量库构建脚本
│
├── GraphRAG/                       # GraphRAG 智能体模块
│   ├── graphrag_agent.py           #   GraphRAG Agent（全局检索+自纠错）
│   ├── create_graphrag_datasets.py #   GraphRAG 数据集构建脚本
│   └── tourist_graphrag/           #   GraphRAG 工作目录
│       ├── settings.yaml           #     GraphRAG 配置（DeepSeek 模型）
│       ├── input/                  #     输入文本数据
│       ├── output/                 #     知识图谱索引输出
│       └── prompts/                #     自定义提示词模板
│
├── vlm/                            # 多模态 VLM RAG 模块
│   ├── vlm_multimodel_rag.py       #   VLM 文档处理管线（PDF→图片→OCR→向量库）
│   ├── multi_model_agent.py        #   多模态 RAG Agent
│   └── result_markdown/            #   VLM 解析后的 Markdown 文档
│       ├── xianggang/              #     香港旅游攻略
│       ├── aomen/                  #     澳门旅游攻略
│       └── taiwan/                 #     台湾旅游攻略
│
├── pure_ocr/                       # 纯 OCR 文档处理模块
│   ├── ocr_multimodel_rag.py       #   OCR 文档处理管线（Tesseract+Unstructured）
│   ├── image_path/                 #   PDF 提取的图片资源
│   └── result_markdown/            #   OCR 解析后的 Markdown 文档
│
├── faiss_index/                    # 预构建的 FAISS 向量索引
│   ├── index.faiss
│   └── index.pkl
│
└── build_system_graph.png          # 系统流程图
```

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **Agent 编排** | LangGraph | 状态图工作流，条件路由，节点编排 |
| **LLM** | DeepSeek (deepseek-chat) | 对话生成、结构化输出、工具调用 |
| **Embedding** | DashScope (text-embedding-v3) | 向量嵌入，通过通义千问 API |
| **向量检索** | FAISS (faiss-cpu) | 本地持久化向量数据库 |
| **知识图谱** | GraphRAG 2.7.0 | 实体抽取、社区聚类、全局/局部检索 |
| **VLM OCR** | olmOCR-7B + vLLM | 视觉语言模型，PDF 页面级 OCR |
| **纯 OCR** | Tesseract + Unstructured | 传统 OCR + 版面分析 + 表格识别 |
| **版面检测** | YOLO-X (Layout) | 文档区域目标检测 |
| **表格识别** | Table Transformer | 表格结构化识别 |
| **PDF 处理** | PyMuPDF (fitz) | PDF 页面渲染、图片提取 |
| **文档分割** | MarkdownHeaderTextSplitter | 按 Markdown 标题层级切分文档 |

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
QWEN_API_KEY=your_qwen_api_key
GRAPHRAG_API_KEY=your_deepseek_api_key
```

### 3. 系统依赖（VLM/OCR 管线）

```bash
# Linux 环境需安装 poppler-utils
sudo apt-get install poppler-utils

# 安装 Tesseract OCR
sudo apt-get install libleptoncia-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn
```

### 4. 模型下载（VLM/OCR 管线）

```bash
# olmOCR-7B 模型（VLM 管线，需 GPU 显存 ≥ 15GB）
modelscope download --model allenai/olmOCR-7B-0725-FP8 --local_dir ./olmOCR-7B-0725-FP8

# 版面检测与表格识别模型（纯 OCR 管线）
modelscope download --model microsoft/table-transformer-structure-recognition --local_dir ./table-transformer-structure-recognition
modelscope download --model timm/resnet18.a1_in1k --local_dir ./resnet18.a1_in1k
modelscope download --model AI-ModelScope/yolo_x_layout --local_dir ./yolo_x_layout
```

## 使用方式

### 启动完整系统

```bash
python System_Agent.py
```

系统启动后进入交互式对话，支持多轮对话：

```
请输入本轮问题（输入 'exit' 退出）:
> 北京有哪些好玩的景点？           → 路由到 rag_agent
> 南昌和浙江饮食文化有什么不同？    → 路由到 graphrag_agent
> 香港有什么好玩的地方？           → 路由到 multi_model_agent
```

### 单独运行各智能体

```bash
# Naive RAG Agent
python -c "from Naive_RAG.rag_agent import rag_agent; ..."

# GraphRAG Agent
python -c "from GraphRAG.graphrag_agent import graphrag_agent; ..."

# 多模态 RAG Agent
python -c "from vlm.multi_model_agent import multi_model_agent; ..."
```

### 构建向量知识库

```bash
# 构建 Naive RAG 的 FAISS 向量库
python Naive_RAG/create_vectorstore.py

# 构建 GraphRAG 知识图谱索引
graphrag init --root ./GraphRAG/tourist_graphrag
graphrag index --root ./GraphRAG/tourist_graphrag

# 构建 VLM 多模态向量库
python vlm/vlm_multimodel_rag.py

# 构建纯 OCR 向量库
python pure_ocr/ocr_multimodel_rag.py
```

## RAG 智能体工作流

每个 RAG 智能体内部遵循相同的高质量检索范式：

```
START
  │
  ▼
generate_response ──→ LLM 决定是否调用检索工具
  │                         │
  │                    (有工具调用)
  ▼                         ▼
retrieve ────────────── 调用检索工具获取知识
  │
  ▼
grade_search_docs ──→ 评估检索结果相关性
  │              │
  │(相关)        │(不相关)
  ▼              ▼
generate_final   refine_query ──→ 改写问题
  │              │                    │
  ▼              └────────────────────┘
  END                        (回到 generate_response 重试)
```

## 查询示例

| 问题类型 | 示例问题 | 路由目标 |
|---------|---------|---------|
| 基础景点咨询 | 从大理坐大巴到双廊需要多久？ | rag_agent |
| 交通线路 | 从郑州出发，坐高铁到安阳怎么走？ | rag_agent |
| 多轮对话 | 北京有什么好玩的？ → 到了住哪里？ | rag_agent |
| 多景点比较 | 南昌和浙江饮食文化有什么不同？ | graphrag_agent |
| 主题分析 | 中国有哪些有名的"山"类型的景点？ | graphrag_agent |
| 出发推荐 | 从上海出发有哪些景点可以去？ | graphrag_agent |
| 港澳台咨询 | 香港有什么好玩的地方？ | multi_model_agent |
| 港澳台攻略 | 台湾旅游需要准备什么？ | multi_model_agent |
| 跨地区对比 | 台湾、香港、澳门旅游有什么差异点？ | multi_model_agent |
