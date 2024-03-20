
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class Report:
    def __init__(self):
        self.language = 'Chinese'

        from dotenv import load_dotenv
        load_dotenv()

        self.llm  = ChatOpenAI()
        # self.llm  = ChatOpenAI(
        # model_name="chatglm3",
        # openai_api_base="http://localhost:8000/v1",
        # openai_api_key="EMPTY",
        # streaming=False,
    # )
        # 初始化Prompt
        self.paper_prompt = ChatPromptTemplate(
        messages = [
            SystemMessagePromptTemplate.from_template(
                "你是一个专业的研究者，擅长从论文的首页介绍部分提取关键信息并以Markdown格式进行总结。"
            ),
            HumanMessagePromptTemplate.from_template(
                '''
                给定一篇论文的介绍内容如下：
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
                ''')
        ]

    )



