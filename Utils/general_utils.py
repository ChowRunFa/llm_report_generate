
import markdown

# 这是你的模型函数，它将Markdown格式的字符串转换为HTML
def markdown_to_html(markdown_text):
    html_output = markdown.markdown(markdown_text)
    return html_output
