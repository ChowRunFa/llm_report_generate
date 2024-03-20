from Object.Paper import Paper
from Utils.general_utils import markdown_to_html
import gradio as gr
import zipfile
import os
from Utils.general_utils import is_zip_file
from Utils.general_utils import is_pdf_file

def report_generate_main(files):
    # 处理单个PDF文件

    markdown_content = Paper(file=files).get_paper_info()

    markdown_content +=  "<div > <img  src='file/extracted_images/max_image.png'> </div>"

    return markdown_to_html(markdown_content)


# 创建Gradio界面
iface = gr.Interface(
    fn=report_generate_main,
    inputs=gr.inputs.File(type='file'),
    outputs="html",
    title="LLM Report Generator",
    description="Upload a PDF file to get a report."
)


if __name__ == '__main__':

    # 启动Gradio应用
    iface.launch()