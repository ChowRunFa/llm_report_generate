import fitz
import time
import re
import os
import gradio as gr

def pdfTOpic(path):
    pic_folder = 'extracted_images'  # 图片保存的文件夹
    if not os.path.exists(pic_folder):
        os.makedirs(pic_folder)

    t0 = time.perf_counter()

    checkXO = r"/Type(?= */XObject)"
    checkIM = r"/Subtype(?= */Image)"

    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"Error: {e}")
        return "Error: Unable to open the PDF file"

    imgCount = 0
    lenXREF = doc.xref_length()

    print("文件名:{},页数:{},对象:{}".format(path, len(doc), lenXREF - 1))

    for i in range(1, lenXREF):
        text = doc.xref_object(i)
        isXObject = re.search(checkXO, text)
        isImage = re.search(checkIM, text)

        if not isXObject or not isImage:
            continue

        imgCount += 1
        pix = fitz.Pixmap(doc, i)
        new_name = os.path.join(pic_folder, path.replace('\\', '_') + f"_img{imgCount}.png")
        new_name = new_name.replace(':', '')

        if pix.n < 5:
            pix.save(new_name)
        else:
            pix0 = fitz.Pixmap(fitz.csRGB, pix)
            pix0.save(new_name)
            pix0 = None

        pix = None

    t1 = time.perf_counter()
    print("运行时间:{}s".format(t1 - t0))
    print("提取了{}张图片".format(imgCount))

    return pic_folder  # 返回图片保存的文件夹路径

iface = gr.Interface(fn=pdfTOpic, inputs="text", outputs="text")
iface.launch()
