import gradio as gr

from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain_community.chat_models.openai import ChatOpenAI

load_dotenv()


# 初始化OpenAI LLM
llm = OpenAI(temperature=0.2)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 解析PDF文件并生成摘要的函数
def parse_pdf(pdf_file):
    TOKEN_LIMIT_PER_FRAGMENT = 2500

    from Utils.split_utils import read_and_clean_pdf_text
    file_content, page_one = read_and_clean_pdf_text(pdf_file) # （尝试）按照章节切割PDF
    file_content = file_content.encode('utf-8', 'ignore').decode()  # avoid reading non-utf8 chars
    page_one = str(page_one).encode('utf-8', 'ignore').decode()  # avoid reading non-utf8 chars

    # 假设每个字符是一个Token（这是一个简化的假设）
    from Utils.split_utils import split_text_to_satisfy_token_limit
    paper_fragments = split_text_to_satisfy_token_limit(txt=file_content, limit=TOKEN_LIMIT_PER_FRAGMENT)
    page_one_fragments = split_text_to_satisfy_token_limit(txt=str(page_one), limit=TOKEN_LIMIT_PER_FRAGMENT // 4)

    paper_meta = page_one_fragments[0].split('introduction')[0].split('Introduction')[0].split('INTRODUCTION')[0]

    final_results = []
    for fragment in paper_fragments:
        prompt = f"中文总结以下文本:\n\n{fragment}"
        gpt_response = llm(prompt)
        final_results.append(gpt_response)

    # 将所有摘要连接起来
    return ' '.join(final_results)


# 创建Gradio界面
iface = gr.Interface(
    fn=parse_pdf,
    inputs=gr.inputs.File(type='file'),
    outputs='text',
    title="PDF Summarizer",
    description="Upload a PDF file to get a summary."
)

# 启动Gradio应用
iface.launch()