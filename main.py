from Object.Paper import Paper
from Utils.general_utils import markdown_to_html
import gradio as gr
import zipfile
import os

def report_generate_main(file_path):
    # 检查文件是否为zip文件
    if zipfile.is_zipfile(file_path):
        # 创建一个临时目录来解压文件
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # 创建临时目录
            temp_dir = "temp_unzip_dir"
            os.makedirs(temp_dir, exist_ok=True)
            zip_ref.extractall(temp_dir)

            # 获取解压后的所有文件路径
            file_paths = [os.path.join(temp_dir, name) for name in zip_ref.namelist()]

            # 为每个文件生成报告
            reports = []
            for file in file_paths:
                p = Paper(file=file)
                reports.append(markdown_to_html(p.paper_info))

            # 清理临时文件和目录
            for file in file_paths:
                os.remove(file)
            os.rmdir(temp_dir)

            # 返回所有报告的HTML
            return "<br>".join(reports)  # 使用<br>标签来分隔不同的报告
    else:
        # 如果不是zip文件，假设它是一个可以直接处理的文件
        p = Paper(file=file_path)
        return markdown_to_html(p.paper_info)


if __name__ == '__main__':
    # 创建Gradio界面
    iface = gr.Interface(
        fn=report_generate_main,
        inputs=gr.inputs.File(type='file'),
        outputs="html",
        title="LLM Report Generator",
        description="Upload a PDF file to get a report."
    )
    # 启动Gradio应用
    iface.launch()