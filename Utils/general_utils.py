import zipfile

import fitz  # PyMuPDF
import markdown

# 这是你的模型函数，它将Markdown格式的字符串转换为HTML
def markdown_to_html(markdown_text):
    html_output = markdown.markdown(markdown_text)
    return html_output

def is_zip_file(file_path):
    return zipfile.is_zipfile(file_path)

def is_pdf_file(file_path):
    try:
        # 尝试打开 PDF 文件
        with fitz.open(file_path) as doc:
            return True  # 如果没有异常抛出，则文件是一个 PDF
    except fitz.fitz.FileDataError:
        return False  # 如果抛出异常，则文件不是一个 PDF