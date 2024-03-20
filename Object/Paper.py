import json
import re
import gradio as gr

import fitz, io, os
from PIL import Image
import dotenv
from langchain_community.chat_models.openai import ChatOpenAI
import openai

dotenv.load_dotenv()
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class Paper:
    def __init__(self,file):
        self.file = file
        self.paper_info = self.get_paper_info()
        self.max_img = self.get_image_path("D:\Pycharm_Projects\llm_report_generate\extracted_images")

    def get_image_path(self, pic_path):
        if not os.path.exists(pic_path):
            os.mkdir(pic_path)
        checkIM = r"/Subtype(?= */Image)"
        with fitz.open(self.file) as doc:
            lenXREF = doc.xref_length()
            max_size = 0
            max_img = None
            for i in range(1, lenXREF):
                text = doc.xref_object(i)
                isImage = re.search(checkIM, text)
                if not isImage:
                    continue
                pix = fitz.Pixmap(doc, i)
                if pix.size < 10000:  # 设置最小图片尺寸阈值
                    continue

                if pix.size > max_size:
                    max_size = pix.size
                    if max_img:
                        max_img.clear_with()  # 删除之前保存的较小图片
                    max_img = pix

            if max_img:
                max_img.save(os.path.join(pic_path, "max_image.png"))
            return os.path.join(pic_path, "max_image.png")

    def get_paper_info(self):

        llm = OpenAI(temperature=.7)
        # llm = ChatOpenAI(
        #     model_name="chatglm",
        #     openai_api_base="http://localhost:8000/v1",
        #     openai_api_key="EMPTY",
        #     streaming=False,
        # )

        template = """
        你是一个专业的研究者，擅长从论文的首页介绍部分提取关键信息并以Markdown格式进行总结。给定一篇论文的介绍内容如下：

        {content}

        请根据以上内容，提取并总结以下信息：

        1. 论文的标题（中英文）
        2. 详细的摘要（中英文）
        3. 第一作者的姓名
        4. 第一作者的所属单位
        5. 论文的发表日期
        6. 论文的出版单位

        并按照以下Markdown格式组织这些信息：

        ```markdown
        # 论文标题
        - 中文标题: (中文标题)
        - English Title: (英文标题)

        ## 主要信息
        - 第一作者: (第一作者姓名)
        - 所属单位: (第一作者所属单位)
        - 发表日期: (发表日期)
        - 出版单位: (出版单位)

        ### English Abstract
        - (英文摘要)

        ### 中文摘要
        - (中文摘要)

        ``` 

        请注意，上面的()是用来指示需要填充的信息，实际使用时需要替换为相应的内容。
        请将实际的内容替换到相应的大括号位置，并确保Markdown格式正确，以便于阅读和理解。
        """

        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(template))

        with fitz.open(self.file) as doc:
            first_page_content = doc[0].get_text()
            response = llm_chain(first_page_content)['text']
            print(response)
            return response

