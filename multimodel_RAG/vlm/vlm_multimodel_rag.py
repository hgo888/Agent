"""
使用vlm模型进行pdf分析测试代码
1.前置环境安装
pip install "olmocr[gpu]"
olmOCR只支持本地部署，硬件条件GPU显存高于15 GB
sudo apt-get update
sudo apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools

2.模型下载
mkdir olmOCR-7B-0725-FP8
modelscope download --model allenai/olmOCR-7B-0725-FP8 --local_dir ./olmOCR-7B-0725-FP8
3.通过vllm启动服务:
vllm serve ./olmOCR-7B-0725-FP8 \
  --served-model-name olmocr \
  --max-model-len 16000 \
  --port 3000

4.测试效果
# Download a sample test PDF
curl -o olmocr-sample.pdf https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf
#调用原生api生成对应的结果
#(1) OCR识别1份文件
先进入文件所在的目录:
cd /root/autodl-tmp/multimodel_RAG/vlm/
python -m olmocr.pipeline ./localworkspace --markdown --model /root/autodl-tmp/olmOCR-7B-0725-FP8   --pdfs ./olmocr-sample.pdf
#(2) OCR识别多份文件
先进入文件所在的目录:
cd /root/autodl-tmp/multimodel_RAG/vlm/
python -m olmocr.pipeline ./localworkspace --markdown --model /root/autodl-tmp/olmOCR-7B-0725-FP8  --pdfs ./*.pdf
"""
#%%
# 下面通过将olmOCR做一些改造，使得其具备VLM模型的能力
"""
通过vllm启动olmocr服务引擎:
vllm serve ./olmOCR-7B-0725-FP8 \
  --served-model-name olmocr \
  --max-model-len 16000 \
  --port 3000

多模态olmOCR处理链路：
将多页pdf文档转成多页png图片->

"""
import base64
import requests
from pdf2image import convert_from_path
from PIL import Image
VLLM_ENDPOINT = "http://localhost:3000/v1/chat/completions"  # 改成你的host
MODEL_NAME = "olmocr"  # 必须与 vLLM 的 --served-model-name 一致
#将单个pdf_file转成若干张image存放到指定路径下
def get_page_images(input_pdf_file,output_image_path,MAX_PAGES=200):
    #将PDF转成images，dpi为精度，显存 vs 精度一般取值200~300够用
    print(f"当前正在处理文件:{input_pdf_file}")
    pages = convert_from_path(input_pdf_file, dpi=200)
    print("pages:",pages)
    #存放每个pdf页面转成图片后的数组
    page_images_list = []
    for i, img in enumerate(pages[:MAX_PAGES], start=1):
        # 将较大的图片压缩到1600*1600像素以内
        max_side = max(img.size)
        if max_side > 1600:
            scale = 1600 / max_side
            img = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
        file_name = file.split("/")[-1].replace(".pdf", "")
        if '香港' in file_name:
            try:
                os.mkdir(fix_url + "xianggang")
            except:
                pass
            file_fold_name = "xianggang/"
        elif '澳门' in file_name:
            try:
                os.mkdir(fix_url + "aomen")
            except:
                pass
            file_fold_name = 'aomen/'
        elif '台北' in file_name:
            try:
                os.mkdir(fix_url + "taiwan")
            except:
                pass
            file_fold_name = 'taiwan/'
        else:
            os.mkdir(fix_url + file_name)
            file_fold_name = file_name
        one_pdf_image_output_path=output_image_path+file_fold_name+f"__page_{i}.png"
        print("one_pdf_image_output_path:",one_pdf_image_output_path)
        img.save(one_pdf_image_output_path, "PNG")
        page_images_list.append(one_pdf_image_output_path)
    print("page_images_list:",page_images_list)
    return page_images_list

# input_pdf_file = "/root/autodl-tmp/multimodel_RAG/vlm/我是驴友-台北旅游攻略.pdf"
# page_images_list=get_page_images(input_pdf_file,output_image_path="/root/autodl-tmp/multimodel_RAG/vlm/result_markdown/")
# print(page_images_list)
#%%
# 构造每页的图片聊天消息并调用 vLLM（OpenAI兼容协议）做整体识别
def ocr_each_page(img_path: str) -> str:
    with open(img_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    #Data URL 是标准方式，让模型知道 "这是一个图像，内容在 URL 中"
    #文件读取 → base64编码 → Data URL格式化 → 发送给模型 → 模型解码处理
    # payload = {"image": <binary_data>}  # 会出错
    # image_url = "http://localhost:8000/temp/image.png" 不可取
    image_url=f"data:image/png;base64,{base64_image}"
    message = [{"type": "text",
         "text":"将页面转换为Markdown格式，遵循人类自然阅读顺序。并遵循以下规则："
                   "(1) 将表格转换成Markdown表格；"
                   "(2) 对于流程图、路线图总结每个流程和路线的概要；"
                   "(3) 对于其他图片总结图片描述的内容或场景的主要内容；"
                   "(4) 输出中切勿乱编或自行补充缺失的内容，请勿提供页眉与页脚"
        },
        {"type": "image_url",
         "image_url": {
         "url": image_url,
         "detail": "auto"},  # ["low","high","auto"]
        },
    ]
    payload = {
        "model": "olmocr",     # 要与 vLLM --served-model-name 一致
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    return_result = requests.post("http://localhost:3000/v1/chat/completions", json=payload, timeout=120)
    return_ocr_result=return_result.json()["choices"][0]["message"]["content"]
    return return_ocr_result


def get_each_image_info_combine_markdown(page_images_list: list,output_markdown_path: str):
    # 逐页解析并合并
    markdown_pages = []
    for page in page_images_list:
        try:
            markdown_pages.append(ocr_each_page(page))
        except Exception as e:
            markdown_pages.append(f"\n\n<!-- ERROR on {page}: {e} -->\n\n")
    full_md = "\n\n\\--- Page Break ---\n\n".join(markdown_pages)
    with open(output_markdown_path, "w", encoding="utf-8") as f:
        f.write(full_md)
    print(f"保存vlm markdown成功: {output_markdown_path}")
    return

# output_markdown_path= "/root/autodl-tmp/multimodel_RAG/vlm/result_markdown/taiwan/taiwan.md"
# get_each_image_info_combine_markdown(page_images_list,output_markdown_path)
#%%
#调用上面定义的方法，开始正式处理港澳台旅游攻略数据
###########真实使用时可能会有N多个文档用如下代码轮询获取不同的markdown##########################
import os
def get_file_paths(folder_path):
    """获取文件夹下所有文件的路径（不包含子目录）"""
    file_paths = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            file_paths.append(item_path)
    return file_paths

# 从datasets/gang_ao_pdf下依次读取pdf文件，完成pdf2img、img2markdown分别保存到对应的文件夹中
folder = '/root/autodl-tmp/multimodel_RAG/datasets/gang_ao_pdf'
fix_url = "/root/autodl-tmp/multimodel_RAG/vlm/result_markdown/"
# fix_url = "/root/autodl-tmp/multimodel_RAG/vlm/result_markdown1/"

files = get_file_paths(folder)
for file in files:
    print(f"当前正在处理文件:{file}")
    page_images_list=get_page_images(input_pdf_file=file,output_image_path=fix_url)
    print(f"将pdf->image得到结果:{page_images_list}")
    file_name=file.split("/")[-1].replace(".pdf","")
    if '香港' in file_name:
        try:
            os.mkdir(fix_url+"xianggang")
        except:
            pass
        file_name="xianggang/"
    elif '澳门' in file_name:
        try:
            os.mkdir(fix_url+"aomen")
        except:
            pass
        file_name='aomen/'
    elif '台北' in file_name:
        try:
            os.mkdir(fix_url+"taiwan")
        except:
            pass
        file_name='taiwan/'
    else:
        os.mkdir(fix_url+file_name)
    doc_output_path=fix_url+file_name+file_name.replace("/","")+'.md'
    print("当前输出的markdown路径为:",doc_output_path)
    get_each_image_info_combine_markdown(page_images_list=page_images_list,output_markdown_path=doc_output_path)
#%%
#接下来就是熟悉的文档切块存储策略，但是我们需要将不同地点的知识做一下隔离
# 然后，为了搭建RAG系统，我们还需要对原始文档进行处理，来创建词向量数据库：
from langchain_openai import OpenAIEmbeddings
import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv('/root/autodl-tmp/multimodel_RAG/.env')
# print(os.getenv("QWEN_API_KEY",""))
embeddings_model = OpenAIEmbeddings(
        api_key=os.getenv("QWEN_API_KEY",""),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="text-embedding-v3",
        check_embedding_ctx_length = False,
)
print(embeddings_model)
#%%
#批量获取港澳台文件夹下的markdown文件
parent_dir="/root/autodl-tmp/multimodel_RAG/vlm/result_markdown"
for file_path in parent_dir:
    # 初始化结果列表
    markdown_list = []
    # 遍历父目录下的所有子文件夹
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            # 检查文件扩展名是否为.md
            if file.endswith('.md'):
                # 获取文件的完整路径
                file_path = os.path.join(root, file)
                markdown_list.append(file_path)
    # 打印结果
    print(f"找到 {len(markdown_list)} 个.md文件")
    for path in markdown_list[:10]:  # 只显示前10个路径
        print(path)

#%%
from langchain_community.docstore.document import Document
# 使用空文档初始化向量库
empty_doc = Document(page_content="", metadata={})
vector_store = FAISS.from_documents([empty_doc], embeddings_model)
#%%
"""
headers_to_split_on:第一个元素：Markdown 标题标记符（#, ##, ### 等）
第二个元素：该层级标题在分割后的元数据中的名称

MarkdownHeaderTextSplitter:它会查找文档中的所有标题
只关注 headers_to_split_on 中指定的标题层级
在每个指定的标题处进行文档分割
未指定的标题层级将被忽略（不会触发分割）
"""
# 定义分割器和标题替换策略
headers_to_split_on = [("#", "Header 1"),("##", "Header 2"),("###","Header 3"),("####","Header 4")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
#将一个文件路径下的所有markdown做切分然后存入向量库
def add_multi_markdown_to_vectordb(markdown_list):
    for file in markdown_list:
        with open(file, "r", encoding="utf-8") as f:
            md_content = f.read()
            print("md_content:",md_content)
        md_header_splits = markdown_splitter.split_text(md_content)
        print("xxxx",md_header_splits[1])
        # 指定向量嵌入的批次大小,批次不能太大不然API会报错
        batch_size = 10
        # 将文档分批添加进向量数据库
        for i in range(0, len(md_header_splits), batch_size):
            batch = md_header_splits[i:i + batch_size]
            vector_store.add_documents(batch)
    # 保存向量库
    vector_store.save_local("/root/autodl-tmp/multimodel_RAG/vlm/result_markdown")
    print("向量库构建成功")
    return vector_store

vector_store=add_multi_markdown_to_vectordb(markdown_list)
print("vector_store:",vector_store)
#%%
#加载持久化的向量库，进行检索
import os
from dotenv import load_dotenv
load_dotenv('/root/autodl-tmp/multimodel_RAG/.env')
embeddings_model = OpenAIEmbeddings(
        api_key=os.getenv("QWEN_API_KEY",""),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="text-embedding-v3",
        check_embedding_ctx_length = False,
)
#读取向量库
multi_model_knowledge_path="/root/autodl-tmp/multimodel_RAG/vlm/result_markdown/"  #"mcp_course_materials_db"
vector_store = FAISS.load_local(
    folder_path=multi_model_knowledge_path,
    embeddings=embeddings_model,
    allow_dangerous_deserialization=True,
)
#%%
multi_model_retriever=vector_store.as_retriever(search_type="similarity", #_score_threshold
                                    search_kwargs={"k": 5}) #"score_threshold":0.3,
search_result=multi_model_retriever.invoke("去台湾玩有哪些方式？")
print(search_result)
#%%

multi_model_retriever=vector_store.as_retriever(search_type="similarity", #_score_threshold
                                    search_kwargs={"k": 5}) #"score_threshold":0.3,
search_result=multi_model_retriever.invoke("太平山顶俯瞰维港夜景主要有些什么？")
print(search_result)

#%%
multi_model_retriever=vector_store.as_retriever(search_type="similarity", #_score_threshold
                                    search_kwargs={"k": 5}) #"score_threshold":0.3,
search_result=multi_model_retriever.invoke("港澳通信证如何做申请？")
print(search_result)
#%%
#用相似度阈值进行检索
multi_model_retriever=vector_store.as_retriever(search_type="similarity_score_threshold", #_score_threshold
                                    search_kwargs={"k": 5,"score_threshold":0.3}) #"score_threshold":0.3,
search_result=multi_model_retriever.invoke("澳门外港客运码头票价多少")
print(search_result)
#%%
#用相似度阈值进行检索先前tesseract解析的pdf中检索不到的问题
multi_model_retriever=vector_store.as_retriever(search_type="similarity_score_threshold", #_score_threshold
                                    search_kwargs={"k": 5,"score_threshold":0.3}) #"score_threshold":0.3,
search_result=multi_model_retriever.invoke("夫妻同游需要提供什么")
print(search_result)
