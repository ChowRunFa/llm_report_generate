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

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    paper_fragments = text_splitter.create_documents([file_content])


    final_results = []
    for fragment in paper_fragments:
        prompt = f"中文总结以下文本:\n\n{fragment}"
        gpt_response = llm(prompt)
        print(gpt_response)
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