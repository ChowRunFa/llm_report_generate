# -*- coding: utf-8 -*-
# @Time    : 2024/2/1 0:15
# @Author  : ChowRunFa
# @File    : abstract_generate.py
# @Software: PyCharm
import argparse
import base64
import configparser
import datetime
import json
import os
import re
from collections import namedtuple

import arxiv
import numpy as np
import openai
import requests
import tenacity
import tiktoken

import fitz, io, os
from PIL import Image

class Paper:
    def __init__(self, path, title='', url='', abs='', authers=[]):
        # 初始化函数，根据pdf路径初始化Paper对象
        self.url = url  # 文章链接
        self.path = path  # pdf路径
        self.section_names = []  # 段落标题
        self.section_texts = {}  # 段落内容
        self.abs = abs
        self.title_page = 0
        if title == '':
            self.pdf = fitz.open(self.path)  # pdf文档
            self.title = self.get_title()
            self.parse_pdf()
        else:
            self.title = title
        self.authers = authers
        self.roman_num = ["I", "II", 'III', "IV", "V", "VI", "VII", "VIII", "IIX", "IX", "X"]
        self.digit_num = [str(d + 1) for d in range(10)]
        self.first_image = ''

    def parse_pdf(self):
        self.pdf = fitz.open(self.path)  # pdf文档
        self.text_list = [page.get_text() for page in self.pdf]
        self.all_text = ' '.join(self.text_list)
        self.section_page_dict = self._get_all_page_index()  # 段落与页码的对应字典
        print("section_page_dict", self.section_page_dict)
        self.section_text_dict = self._get_all_page()  # 段落与内容的对应字典
        self.section_text_dict.update({"title": self.title})
        self.section_text_dict.update({"paper_info": self.get_paper_info()})
        self.pdf.close()

    def get_paper_info(self):
        first_page_text = self.pdf[self.title_page].get_text()
        if "Abstract" in self.section_text_dict.keys():
            abstract_text = self.section_text_dict['Abstract']
        else:
            abstract_text = self.abs
        first_page_text = first_page_text.replace(abstract_text, "")
        return first_page_text

    def get_image_path(self, image_path=''):
        """
        将PDF中的第一张图保存到image.png里面，存到本地目录，返回文件名称，供gitee读取
        :param filename: 图片所在路径，"C:\\Users\\Administrator\\Desktop\\nwd.pdf"
        :param image_path: 图片提取后的保存路径
        :return:
        """
        # open file
        max_size = 0
        image_list = []
        with fitz.Document(self.path) as my_pdf_file:
            # 遍历所有页面
            for page_number in range(1, len(my_pdf_file) + 1):
                # 查看独立页面
                page = my_pdf_file[page_number - 1]
                # 查看当前页所有图片
                images = page.get_images()
                # 遍历当前页面所有图片
                for image_number, image in enumerate(page.get_images(), start=1):
                    # 访问图片xref
                    xref_value = image[0]
                    # 提取图片信息
                    base_image = my_pdf_file.extract_image(xref_value)
                    # 访问图片
                    image_bytes = base_image["image"]
                    # 获取图片扩展名
                    ext = base_image["ext"]
                    # 加载图片
                    image = Image.open(io.BytesIO(image_bytes))
                    image_size = image.size[0] * image.size[1]
                    if image_size > max_size:
                        max_size = image_size
                    image_list.append(image)
        for image in image_list:
            image_size = image.size[0] * image.size[1]
            if image_size == max_size:
                image_name = f"image.{ext}"
                im_path = os.path.join(image_path, image_name)
                print("im_path:", im_path)

                max_pix = 480
                origin_min_pix = min(image.size[0], image.size[1])

                if image.size[0] > image.size[1]:
                    min_pix = int(image.size[1] * (max_pix / image.size[0]))
                    newsize = (max_pix, min_pix)
                else:
                    min_pix = int(image.size[0] * (max_pix / image.size[1]))
                    newsize = (min_pix, max_pix)
                image = image.resize(newsize)

                image.save(open(im_path, "wb"))
                return im_path, ext
        return None, None

    # 定义一个函数，根据字体的大小，识别每个章节名称，并返回一个列表
    def get_chapter_names(self, ):
        # # 打开一个pdf文件
        doc = fitz.open(self.path)  # pdf文档
        text_list = [page.get_text() for page in doc]
        all_text = ''
        for text in text_list:
            all_text += text
        # # 创建一个空列表，用于存储章节名称
        chapter_names = []
        for line in all_text.split('\n'):
            line_list = line.split(' ')
            if '.' in line:
                point_split_list = line.split('.')
                space_split_list = line.split(' ')
                if 1 < len(space_split_list) < 5:
                    if 1 < len(point_split_list) < 5 and (
                            point_split_list[0] in self.roman_num or point_split_list[0] in self.digit_num):
                        print("line:", line)
                        chapter_names.append(line)
                        # 这段代码可能会有新的bug，本意是为了消除"Introduction"的问题的！
                    elif 1 < len(point_split_list) < 5:
                        print("line:", line)
                        chapter_names.append(line)

        return chapter_names

    def get_title(self):
        doc = self.pdf  # 打开pdf文件
        max_font_size = 0  # 初始化最大字体大小为0
        max_string = ""  # 初始化最大字体大小对应的字符串为空
        max_font_sizes = [0]
        for page_index, page in enumerate(doc):  # 遍历每一页
            text = page.get_text("dict")  # 获取页面上的文本信息
            blocks = text["blocks"]  # 获取文本块列表
            for block in blocks:  # 遍历每个文本块
                if block["type"] == 0 and len(block['lines']):  # 如果是文字类型
                    if len(block["lines"][0]["spans"]):
                        font_size = block["lines"][0]["spans"][0]["size"]  # 获取第一行第一段文字的字体大小
                        max_font_sizes.append(font_size)
                        if font_size > max_font_size:  # 如果字体大小大于当前最大值
                            max_font_size = font_size  # 更新最大值
                            max_string = block["lines"][0]["spans"][0]["text"]  # 更新最大值对应的字符串
        max_font_sizes.sort()
        print("max_font_sizes", max_font_sizes[-10:])
        cur_title = ''
        for page_index, page in enumerate(doc):  # 遍历每一页
            text = page.get_text("dict")  # 获取页面上的文本信息
            blocks = text["blocks"]  # 获取文本块列表
            for block in blocks:  # 遍历每个文本块
                if block["type"] == 0 and len(block['lines']):  # 如果是文字类型
                    if len(block["lines"][0]["spans"]):
                        cur_string = block["lines"][0]["spans"][0]["text"]  # 更新最大值对应的字符串
                        font_flags = block["lines"][0]["spans"][0]["flags"]  # 获取第一行第一段文字的字体特征
                        font_size = block["lines"][0]["spans"][0]["size"]  # 获取第一行第一段文字的字体大小
                        # print(font_size)
                        if abs(font_size - max_font_sizes[-1]) < 0.3 or abs(font_size - max_font_sizes[-2]) < 0.3:
                            # print("The string is bold.", max_string, "font_size:", font_size, "font_flags:", font_flags)
                            if len(cur_string) > 4 and "arXiv" not in cur_string:
                                # print("The string is bold.", max_string, "font_size:", font_size, "font_flags:", font_flags)
                                if cur_title == '':
                                    cur_title += cur_string
                                else:
                                    cur_title += ' ' + cur_string
                            self.title_page = page_index
                            # break
        title = cur_title.replace('\n', ' ')
        return title

    def _get_all_page_index(self):
        # 定义需要寻找的章节名称列表
        section_list = ["Abstract",
                        'Introduction', 'Related Work', 'Background',
                        "Preliminary", "Problem Formulation",
                        'Methods', 'Methodology', "Method", 'Approach', 'Approaches',
                        # exp
                        "Materials and Methods", "Experiment Settings",
                        'Experiment', "Experimental Results", "Evaluation", "Experiments",
                        "Results", 'Findings', 'Data Analysis',
                        "Discussion", "Results and Discussion", "Conclusion",
                        'References']
        # 初始化一个字典来存储找到的章节和它们在文档中出现的页码
        section_page_dict = {}
        # 遍历每一页文档
        for page_index, page in enumerate(self.pdf):
            # 获取当前页面的文本内容
            cur_text = page.get_text()
            # 遍历需要寻找的章节名称列表
            for section_name in section_list:
                # 将章节名称转换成大写形式
                section_name_upper = section_name.upper()
                # 如果当前页面包含"Abstract"这个关键词
                if "Abstract" == section_name and section_name in cur_text:
                    # 将"Abstract"和它所在的页码加入字典中
                    section_page_dict[section_name] = page_index
                # 如果当前页面包含章节名称，则将章节名称和它所在的页码加入字典中
                else:
                    if section_name + '\n' in cur_text:
                        section_page_dict[section_name] = page_index
                    elif section_name_upper + '\n' in cur_text:
                        section_page_dict[section_name] = page_index
        # 返回所有找到的章节名称及它们在文档中出现的页码
        return section_page_dict

    def _get_all_page(self):
        """
        获取PDF文件中每个页面的文本信息，并将文本信息按照章节组织成字典返回。

        Returns:
            section_dict (dict): 每个章节的文本信息字典，key为章节名，value为章节文本。
        """
        text = ''
        text_list = []
        section_dict = {}

        # 再处理其他章节：
        text_list = [page.get_text() for page in self.pdf]
        for sec_index, sec_name in enumerate(self.section_page_dict):
            print(sec_index, sec_name, self.section_page_dict[sec_name])
            if sec_index <= 0 and self.abs:
                continue
            else:
                # 直接考虑后面的内容：
                start_page = self.section_page_dict[sec_name]
                if sec_index < len(list(self.section_page_dict.keys())) - 1:
                    end_page = self.section_page_dict[list(self.section_page_dict.keys())[sec_index + 1]]
                else:
                    end_page = len(text_list)
                print("start_page, end_page:", start_page, end_page)
                cur_sec_text = ''
                if end_page - start_page == 0:
                    if sec_index < len(list(self.section_page_dict.keys())) - 1:
                        next_sec = list(self.section_page_dict.keys())[sec_index + 1]
                        if text_list[start_page].find(sec_name) == -1:
                            start_i = text_list[start_page].find(sec_name.upper())
                        else:
                            start_i = text_list[start_page].find(sec_name)
                        if text_list[start_page].find(next_sec) == -1:
                            end_i = text_list[start_page].find(next_sec.upper())
                        else:
                            end_i = text_list[start_page].find(next_sec)
                        cur_sec_text += text_list[start_page][start_i:end_i]
                else:
                    for page_i in range(start_page, end_page):
                        #                         print("page_i:", page_i)
                        if page_i == start_page:
                            if text_list[start_page].find(sec_name) == -1:
                                start_i = text_list[start_page].find(sec_name.upper())
                            else:
                                start_i = text_list[start_page].find(sec_name)
                            cur_sec_text += text_list[page_i][start_i:]
                        elif page_i < end_page:
                            cur_sec_text += text_list[page_i]
                        elif page_i == end_page:
                            if sec_index < len(list(self.section_page_dict.keys())) - 1:
                                next_sec = list(self.section_page_dict.keys())[sec_index + 1]
                                if text_list[start_page].find(next_sec) == -1:
                                    end_i = text_list[start_page].find(next_sec.upper())
                                else:
                                    end_i = text_list[start_page].find(next_sec)
                                cur_sec_text += text_list[page_i][:end_i]
                section_dict[sec_name] = cur_sec_text.replace('-\n', '').replace('\n', ' ')
        return section_dict

# 定义Report类
class Report:
    # 初始化方法，设置属性
    def __init__(self,root_path='./',):
        self.language = 'Chinese'
        self.root_path = root_path
        # 创建一个ConfigParser对象
        self.config = configparser.ConfigParser()
        # 读取配置文件
        self.config.read('apikey.ini')
        OPENAI_KEY = os.environ.get("OPENAI_KEY", "")
        # 获取某个键对应的值
        openai.api_base = self.config.get('OpenAI', 'OPENAI_API_BASE')
        self.chat_api_list = self.config.get('OpenAI', 'OPENAI_API_KEYS')[1:-1].replace('\'', '').split(',')
        self.chat_api_list.append(OPENAI_KEY)

        # prevent short strings from being incorrectly used as API keys.
        self.chat_api_list = [api.strip() for api in self.chat_api_list if len(api) > 20]
        self.chatgpt_model = self.config.get('OpenAI', 'CHATGPT_MODEL')

        self.cur_api = 0
        self.file_format = 'md'
        self.max_token_num = 4096
        self.encoding = tiktoken.get_encoding("gpt2")

    def news_with_chat(self,news_list):
        htmls = []
        for paper_index, paper in enumerate(news_list):
            # 第一步先用title，abs，和introduction进行总结。
            text = ''
            text += 'Title:' + paper.title
            text += 'Url:' + paper.url
            text += 'Abstract:' + paper.abs
            text += 'Paper_info:' + paper.section_text_dict['paper_info']

            # intro
            text += list(paper.section_text_dict.values())[0]
            chat_summary_text = ""
            try:
                chat_summary_text = self.news_summary(text=text)
            except Exception as e:
                print("summary_error:", e)
                import sys
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                if "maximum context" in str(e):
                    current_tokens_index = str(e).find("your messages resulted in") + len(
                        "your messages resulted in") + 1
                    offset = int(str(e)[current_tokens_index:current_tokens_index + 4])
                    summary_prompt_token = offset + 1000 + 150
                    chat_summary_text = self.news_summary(text=text, summary_prompt_token=summary_prompt_token)

            htmls.append('## News:' + str(paper_index + 1))
            htmls.append('\n\n\n')
            htmls.append(chat_summary_text)

            # # 整合成一个文件，打包保存下来。
            date_str = str(datetime.datetime.now())[:13].replace(' ', '-')
            export_path = os.path.join(self.root_path, 'export')
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            mode = 'w' if paper_index == 0 else 'a'
            file_name = os.path.join(export_path,
                                     date_str + '-' + self.validateTitle(paper.title[:80]) + "." + self.file_format)
            self.export_to_markdown("\n".join(htmls), file_name=file_name, mode=mode)

            # file_name = os.path.join(export_path, date_str+'-'+self.validateTitle(paper.title)+".md")
            # self.export_to_markdown("\n".join(htmls), file_name=file_name, mode=mode)
            htmls = []

    def summary_with_chat(self, paper_list):
        htmls = []
        for paper_index, paper in enumerate(paper_list):
            # 第一步先用title，abs，和introduction进行总结。
            text = ''
            text += 'Title:' + paper.title
            text += 'Url:' + paper.url
            text += 'Abstract:' + paper.abs
            text += 'Paper_info:' + paper.section_text_dict['paper_info']

            # intro
            text += list(paper.section_text_dict.values())[0]
            chat_summary_text = ""
            try:
                chat_summary_text = self.chat_summary(text=text)
            except Exception as e:
                print("summary_error:", e)
                import sys
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                if "maximum context" in str(e):
                    current_tokens_index = str(e).find("your messages resulted in") + len(
                        "your messages resulted in") + 1
                    offset = int(str(e)[current_tokens_index:current_tokens_index + 4])
                    summary_prompt_token = offset + 1000 + 150
                    chat_summary_text = self.chat_summary(text=text, summary_prompt_token=summary_prompt_token)

            htmls.append('## Paper:' + str(paper_index + 1))
            htmls.append('\n\n\n')
            htmls.append(chat_summary_text)

            # # 整合成一个文件，打包保存下来。
            date_str = str(datetime.datetime.now())[:13].replace(' ', '-')
            export_path = os.path.join(self.root_path, 'export')
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            mode = 'w' if paper_index == 0 else 'a'
            file_name = os.path.join(export_path,
                                     date_str + '-' + self.validateTitle(paper.title[:80]) + "." + self.file_format)
            self.export_to_markdown("\n".join(htmls), file_name=file_name, mode=mode)

            # file_name = os.path.join(export_path, date_str+'-'+self.validateTitle(paper.title)+".md")
            # self.export_to_markdown("\n".join(htmls), file_name=file_name, mode=mode)
            htmls = []

    def  patents_with_chat(self, patents_list):
        htmls = []
        for paper_index, paper in enumerate(patents_list):
            # 第一步先用title，abs，和introduction进行总结。
            text = ''
            text += 'Title:' + paper.title
            text += 'Url:' + paper.url
            text += 'Abstract:' + paper.abs
            text += 'Paper_info:' + paper.section_text_dict['paper_info']

            # intro
            text += list(paper.section_text_dict.values())[0]
            chat_summary_text = ""
            try:
                chat_summary_text = self.patents_summary(text=text)
            except Exception as e:
                print("summary_error:", e)
                import sys
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                if "maximum context" in str(e):
                    current_tokens_index = str(e).find("your messages resulted in") + len(
                        "your messages resulted in") + 1
                    offset = int(str(e)[current_tokens_index:current_tokens_index + 4])
                    summary_prompt_token = offset + 1000 + 150
                    chat_summary_text = self.patents_summary(text=text, summary_prompt_token=summary_prompt_token)

            htmls.append('## Patent:' + str(paper_index + 1))
            htmls.append('\n\n\n')
            htmls.append(chat_summary_text)

            # # 整合成一个文件，打包保存下来。
            date_str = str(datetime.datetime.now())[:13].replace(' ', '-')
            export_path = os.path.join(self.root_path, 'export')
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            mode = 'w' if paper_index == 0 else 'a'
            file_name = os.path.join(export_path,
                                     date_str + '-' + self.validateTitle(paper.title[:80]) + "." + self.file_format)
            self.export_to_markdown("\n".join(htmls), file_name=file_name, mode=mode)

            # file_name = os.path.join(export_path, date_str+'-'+self.validateTitle(paper.title)+".md")
            # self.export_to_markdown("\n".join(htmls), file_name=file_name, mode=mode)
            htmls = []

    def validateTitle(self, title):
        # 将论文的乱七八糟的路径格式修正
        rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
        new_title = re.sub(rstr, "_", title)  # 替换为下划线
        return new_title

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_summary(self, text, summary_prompt_token=1100):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = 0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(len(text) * (self.max_token_num - summary_prompt_token) / text_token)
        clip_text = text[:clip_text_index]
        messages = [
            {"role": "system",
             "content": "You are a researcher in the field of {}  who is good at summarizing papers using concise statements"},
            {"role": "assistant",
             "content": "This is the title, author, link, abstract and introduction of an English document. I need your help to read and summarize the following questions: " + clip_text},
            {"role": "user", "content": """                 
                 1. Mark the title of the paper (with Chinese translation)
                 2. Mark the title of the paper (use English)
                 3. list all the authors' names (use English)
                 4. mark the first author's affiliation (总结这篇文章的发布单位和发布时间)     
                 5. summarize according to the following four points.Be sure to use {} answers (proper nouns need to be marked in English)
                    - (1):What is the research background of this article?
                    - (2):What are the past methods? What are the problems with them? Is the approach well motivated?
                    - (3):What is the research methodology proposed in this paper?
                    - (4):On what task and what performance is achieved by the methods in this paper? Can the performance support their goals?
                 Follow the format of the output that follows:                  
                 1. 标题 : xxx\n\n
                 2. Title : xxx\n\n
                 3. xxxx年xx月，由xxxx等在xxxx上发表\n\n                       
                 4. Summary: \n\n

                 Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not have too much repetitive information, numerical values using the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed.                 
                 """.format(self.language, self.language, self.language)},
        ]

        if openai.api_type == 'azure':
            response = openai.ChatCompletion.create(
                engine=self.chatgpt_model,
                # prompt需要用英语替换，少占用token。
                messages=messages,
            )
        else:
            response = openai.ChatCompletion.create(
                model=self.chatgpt_model,
                # prompt需要用英语替换，少占用token。
                messages=messages,
            )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        print("summary_result:\n", result)
        print("prompt_token_used:", response.usage.prompt_tokens,
              "completion_token_used:", response.usage.completion_tokens,
              "total_token_used:", response.usage.total_tokens)
        print("response_time:", response.response_ms / 1000.0, 's')
        return result

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def news_summary(self, text, summary_prompt_token=1100):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = 0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(len(text) * (self.max_token_num - summary_prompt_token) / text_token)
        clip_text = text[:clip_text_index]
        messages = [
            {"role": "system",
             "content": "You are a professional who is good at summarizing news content "},
            {"role": "assistant",
             "content": "This is the introduction of an Chinese news. I need your help to read and summarize the following questions: " + clip_text},
            {"role": "user", "content": """                 
                 1. Mark the title of the news (with Chinese )\n\n
                 2. Summarize the content of the news (with Chinese ): xxx\n\n
                
                 Follow the format of the output that follows:                  
                 1. xxxx年xx月xx日，由xxxx发布了xxxxx —— xxxxx\n\n
                 2. 总结: \n\n
                 Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not have too much repetitive information, numerical values using the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed.                 
                 """.format(self.language, self.language, self.language)},
        ]

        if openai.api_type == 'azure':
            response = openai.ChatCompletion.create(
                engine=self.chatgpt_model,
                # prompt需要用英语替换，少占用token。
                messages=messages,
            )
        else:
            response = openai.ChatCompletion.create(
                model=self.chatgpt_model,
                # prompt需要用英语替换，少占用token。
                messages=messages,
            )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        print("summary_result:\n", result)
        print("prompt_token_used:", response.usage.prompt_tokens,
              "completion_token_used:", response.usage.completion_tokens,
              "total_token_used:", response.usage.total_tokens)
        print("response_time:", response.response_ms / 1000.0, 's')
        return result

    def patents_summary(self, text, summary_prompt_token=1100):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = 0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(len(text) * (self.max_token_num - summary_prompt_token) / text_token)
        clip_text = text[:clip_text_index]
        messages = [
            {"role": "system",
             "content": "You are a professional who is good at summarizing the the publicity information of invention patents in the form of a news statement"},
            {"role": "assistant",
             "content": "This is the introduction of an English document. I need your help to read and summarize the following questions: " + clip_text},
            {"role": "user", "content": """                 
                 1. Mark “xxxxxxx获得了xxxxx专利的授权”或者“xxxxxx机构申请了xxxxx专利”\n\n
                 2. Summarize the content of the patents (with Chinese  in the form of a news statement): xxx\n\n
                 Follow the format of the output that follows:                  
                 1. xxxx年xx月xx日，xxxxx 在 某某国家 申请 or 获得“xxxxxx”专利\n\n
                 2. Summary(with Chinese ): \n\n
                 Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not have too much repetitive information, numerical values using the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed.                 
                 """.format(self.language, self.language, self.language)},
        ]

        if openai.api_type == 'azure':
            response = openai.ChatCompletion.create(
                engine=self.chatgpt_model,
                # prompt需要用英语替换，少占用token。
                messages=messages,
            )
        else:
            response = openai.ChatCompletion.create(
                model=self.chatgpt_model,
                # prompt需要用英语替换，少占用token。
                messages=messages,
            )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        print("summary_result:\n", result)
        print("prompt_token_used:", response.usage.prompt_tokens,
              "completion_token_used:", response.usage.completion_tokens,
              "total_token_used:", response.usage.total_tokens)
        print("response_time:", response.response_ms / 1000.0, 's')
        return result

    def export_to_markdown(self, text, file_name, mode='w'):
        # 使用markdown模块的convert方法，将文本转换为html格式
        # html = markdown.markdown(text)
        # 打开一个文件，以写入模式
        with open(file_name, mode, encoding="utf-8") as f:
            # 将html格式的内容写入文件
            f.write(text)

            # 定义一个方法，打印出读者信息


def report_generate_main(pdf_path,news_path,patents_path):
        reader1 = Report()
        # 开始判断是路径还是文件：
        paper_list = []
        news_list = []
        patents_list = []
        if pdf_path.endswith(".pdf"):
            paper_list.append(Paper(path=pdf_path))
        else:
            for root, dirs, files in os.walk(pdf_path):
                print("root:", root, "dirs:", dirs, 'files:', files)  # 当前目录路径
                for filename in files:
                    # 如果找到PDF文件，则将其复制到目标文件夹中
                    if filename.endswith(".pdf"):
                        paper_list.append(Paper(path=os.path.join(root, filename)))
        if news_path.endswith(".pdf"):
            news_list.append(Paper(path=news_path))
        else:
            for root, dirs, files in os.walk(news_path):
                print("root:", root, "dirs:", dirs, 'files:', files)  # 当前目录路径
                for filename in files:
                    # 如果找到PDF文件，则将其复制到目标文件夹中
                    if filename.endswith(".pdf"):
                        news_list.append(Paper(path=os.path.join(root, filename)))
        if patents_path.endswith(".pdf"):
            news_list.append(Paper(path=patents_path))
        else:
            for root, dirs, files in os.walk(patents_path):
                print("root:", root, "dirs:", dirs, 'files:', files)  # 当前目录路径
                for filename in files:
                    # 如果找到PDF文件，则将其复制到目标文件夹中
                    if filename.endswith(".pdf"):
                        patents_list.append(Paper(path=os.path.join(root, filename)))
        print("------------------paper_num: {}------------------".format(len(paper_list)))
        [print(paper_index, paper_name.path.split('\\')[-1]) for paper_index, paper_name in enumerate(paper_list)]
        reader1.summary_with_chat(paper_list=paper_list)
        reader1.news_with_chat(news_list=news_list)
        reader1.patents_with_chat(patents_list=patents_list)

if __name__ == '__main__':
    import time
    start_time = time.time()
    report_generate_main('D:\Postman\Git\Repo\ChatPaper\datas',r'D:\Pycharm_Projects\llm_report_generate\data\news','D:\Postman\Git\Repo\ChatPaper\data_patent')
    print("summary time:", time.time() - start_time)