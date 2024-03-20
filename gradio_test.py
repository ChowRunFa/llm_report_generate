from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import gradio as gr

app = FastAPI()

HELLO_ROUTE = "/hello"
GOODBYE_ROUTE = "/goodbye"
iframe_dimensions = "height=50% width=100%"

index_html = f'''

<div>
<iframe src={HELLO_ROUTE} {iframe_dimensions}></iframe>
</div>

<div>
<iframe src={GOODBYE_ROUTE} {iframe_dimensions}></iframe>
</div>

'''

@app.get("/", response_class=HTMLResponse)
def index():
    return index_html


def report_generate_main(files):
    # 处理单个PDF文件

    markdown_content = Paper(file=files).get_paper_info()

    markdown_content +=  "<div > <img  src='file/extracted_images/max_image.png'> </div>"


    return markdown_to_html(markdown_content)

# 定义第二个函数，它接受文本作为输入，并返回处理后的文本
def process_text(text):
    # 对文本进行处理的逻辑
    # 假设我们简单地将文本转换为大写
    processed_text = text.upper()
    return processed_text


# 创建Gradio界面
iface1 = gr.Interface(
    fn=report_generate_main,
    inputs=gr.inputs.File(type='file'),
    outputs="html",
    title="LLM Report Generator",
    description="Upload a PDF file to get a report."
)


# 创建第二个Gradio接口
iface2 = gr.Interface(
    fn=process_text,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Type something here..."),
    outputs="text",
    title="Text Processor",
    description="Enter some text to process."
)


app = gr.mount_gradio_app(app, iface1, path=HELLO_ROUTE)
app = gr.mount_gradio_app(app, iface2, path=GOODBYE_ROUTE)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)

