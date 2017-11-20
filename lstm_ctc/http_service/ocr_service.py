from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# -*- coding:utf-8 -*-
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
import Queue
import time
import json
import logging
from . import global_val as global_val
from . import http_handler

define("port", default=8001, help="run on the given port", type=int)

class Application:
    def __init__(self, port=8000, queueSize=10):
        tornado.options.parse_command_line()
        self.queueSize = queueSize
        self.port = port
        self.OCRService = None
        self.initGlobalVals()
        self.initTornadoSetting()

    def initTornadoSetting(self):
        handlers = [
            (r"/file" , http_handler.FileUploadHandler),
        ]
        app = tornado.web.Application(handlers=handlers, debug = True)
        self.httpServer = tornado.httpserver.HTTPServer(app)
        self.httpServer.listen(self.port)

    def initGlobalVals(self):
        global_val.setJobQueue(Queue.Queue(self.queueSize))
        global_val.setRetQueue(Queue.Queue(self.queueSize))

    def initLoggingSetting(self):
        logging.basicConfig(
            level       = logging.INFO,
            format      = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
            datefmt     = '[%y-%m-%d %H:%M:%S]',
        )
    # main loop
    def Start(self):
        if None != self.httpServer:
            tornado.ioloop.IOLoop.instance().start()



