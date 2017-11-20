# -*- coding:utf-8 -*-
import tornado.web
import global_val
import json
import logging
from urllib import quote
import os


class FileUploadHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('''
<html>
  <head><title>Upload File</title></head>
  <body>
    <form action='file' enctype="multipart/form-data" method='post'>
    <input type='file' name='file'/><br/>
    <input type='submit' value='submit'/>
    </form>
  </body>
</html>
''')

    def post(self):
        ret = {'result': 'OK'}
        upload_path = os.path.join(os.path.dirname(__file__), 'files')  # 文件的暂存路径
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        file_metas = self.request.files.get('file', None)  # 提取表单中‘name’为‘file’的文件元数据
        if not file_metas:
            ret['result'] = 'Invalid Args'
            return ret
        for meta in file_metas:
            filename = meta['filename']
            file_path = os.path.join(upload_path, filename)
            with open(file_path, 'wb') as up:
                up.write(meta['body'])
                global_val.putJob({"username": "xuye", "file_path": file_path})
                # OR do other thing
        ret = global_val.getRet(block=True)
        os.remove(file_path)
        self.write(json.dumps(ret))
