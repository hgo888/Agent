"""
创建用于graphrag检索的数据集：travel_guide.txt(国内旅游概要攻略)和foreign_guide.txt(国外旅游概要攻略)
"""
#%%
import pandas as pd
travel_dataframe=pd.read_csv("/root/autodl-tmp/multimodel_RAG/datasets/travel_guide.csv",encoding="gbk")
print(travel_dataframe)
whole_paragraph_text=""
for index, row in travel_dataframe.iterrows():
    # 构建输出字符串
    output = (
        f"{row['目的地']}旅游攻略："
        f"交通安排：{row['交通安排']}"
        f"住宿推荐：{row['住宿推荐']}"
        f"必打卡景点：{row['必打卡景点']}"
        f"美食推荐：{row['美食推荐']}"
        f"实用小贴士：{row['实用小贴士']}"
        f"旅行感悟：{row['旅行感悟']}\n"
    )
    print(output)
    whole_paragraph_text+=output
    print("\n" + "-"*50 + "\n")

#%%
print(whole_paragraph_text)
#%%
#将whole_paragraph_text的文本保存到txt文件
with open("/root/autodl-tmp/multimodel_RAG/datasets/travel_guide.txt","w",encoding="utf-8") as f:
    f.write(whole_paragraph_text)
#%%
"""
pip install graphrag==2.7.0
(1) 创建graphrag目录
cd /root/autodl-tmp/multimodel_RAG/GraphRAG
mkdir  ./tourist_graphrag
(2) 初始化graphrag配置
graphrag init --root ./tourist_graphrag
cd tourist_graphrag
vi settings.yaml
(4) 修改graphrag的配置文件将其配置为国内模型，vi. settings.yaml
(5) 创建input目录，并将数据文件copy到input目录下
cd ./tourist_graphrag
mkdir input
cp /root/autodl-tmp/multimodel_RAG/datasets/travel_guide.txt  ./input
cd..  cd .. #退回两级目录到/root/autodl-tmp/multimodel_RAG/GraphRAG
#构建graphrag 索引,耐心等待等一段时间完成知识图谱及GraphRAG索引构建
graphrag index --root ./tourist_graphrag
"""
#%%
"""
- 利用graphrag索引进行查询
(0) 前置重要步骤: 修改graphrag源码的提示词使得其能更好的按照json格式化输出:
    - 找到graphrag的安装路径然后进入
       pip show graphrag
       cd /root/miniconda3/lib/python3.12/site-packages/graphrag
    - 接着进入global_search的目录
       cd query/structured_search/global_search
       vi search.py 将源码的_map_response_single_batch方法中的search_messages = [{"role": "system", "content": search_prompt}]
    做如下替换:
    search_messages=[{"role": "system", "content": search_prompt}, {"role": "user", "content": query+'\n\nThe response should be JSON formatted as follows:\n{\n    \"points\": [\n        {\"description\": \"Description of point 1 [Data: Reports (report ids)]\", \"score\": score_value},\n        {\"description\": \"Description of point 2 [Data: Reports (report ids)]\", \"score\": score_value}\n    ]\n'}, ]
    (目的是：加强检索得到的结果能经过LLM返回Json格式供后续代码能解析，部分模型较差的会导致输出不为json进而引发检索失败的问题。)
（1）命令行检索:
global search: graphrag query --root ./tourist_graphrag --method global --query "长沙出发旅游有什么好玩的景点推荐？" --streaming
local search: graphrag query --root ./tourist_graphrag --method local --query "平遥古城位于哪里？" --streaming
drift search: graphrag query --root ./tourist_graphrag --method drift --query "南浔古镇位于哪里？" --streaming
basic search: graphrag query --root ./tourist_graphrag --method basic --query "平遥古城位于哪里？"
"""
#%%
# （2）graphrag与python做交互
from graphrag.config.load_config import load_config
import pandas as pd
import graphrag.api as api
from pathlib import Path
cfg = load_config(Path("/root/autodl-tmp/multimodel_RAG/GraphRAG/tourist_graphrag/"))
print(cfg)
#%%
entities = pd.read_parquet("/root/autodl-tmp/multimodel_RAG/GraphRAG/tourist_graphrag/output/entities.parquet")
communities = pd.read_parquet("/root/autodl-tmp/multimodel_RAG/GraphRAG/tourist_graphrag/output/communities.parquet")
community_reports = pd.read_parquet("/root/autodl-tmp/multimodel_RAG/GraphRAG/tourist_graphrag/output/community_reports.parquet")
text_units = pd.read_parquet("/root/autodl-tmp/multimodel_RAG/GraphRAG/tourist_graphrag/output/text_units.parquet")
relationships = pd.read_parquet("/root/autodl-tmp/multimodel_RAG/GraphRAG/tourist_graphrag/output/relationships.parquet")
print("entities:",len(entities)) #4248
print("communities:",len(communities)) #787
print("community_reports:",len(community_reports)) #787
print("text_units:",len(text_units)) #563
#%%
#local检索
local_query="南浔古镇在哪里"
local_response, context_data = await api.local_search(
    config=cfg,
    entities=entities,
    communities=communities,
    community_reports=community_reports,
    text_units=text_units,
    relationships=relationships,
    covariates=None,
    community_level=3,
    response_type="Multi-Page Report",
    #Multiple Paragraphs：LLM会生成一个详尽但相对简洁的回答/Multi-Page Report：LLM会生成一个结构化、详细、长度显著增加的正式文档
    query=local_query,
    callbacks=None,
    verbose=True
)
print("local_search结果:",local_response)
#%%
#global检索
global_query="从上海出发有哪些景点可以去"
global_response, context_data1 = await api.global_search(
    config=cfg,
    entities=entities,
    communities=communities,
    community_reports=community_reports,
    community_level=2, #社区划分的层级深度,值越大表示更细粒度的社区划分,level=2 表示中等粒度的社区结构
    dynamic_community_selection=False, #False
    query=global_query,
)
print("global_search结果:",global_response)
#%%