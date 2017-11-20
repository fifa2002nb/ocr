# -*- coding:utf-8 -*-
'''
全局变量文件
'''
import Queue
import thread

#取消ssl自签名验证
#ssl._create_default_https_context = ssl._create_unverified_context  

class GlobalVal:
	#登陆任务队列，线程安全不需要锁
    jobQueue 	= None
    retQueue    = None

# job queue ops
def setJobQueue(jobQueue):
	GlobalVal.jobQueue = jobQueue

def getJobQueue():
    return GlobalVal.jobQueue

def putJob(item, block = True):
    if None == GlobalVal.jobQueue:
	    setJobQueue(Queue.Queue(maxsize = 100))
    GlobalVal.jobQueue.put(item, block)

def getJob(block = True):
    if None == GlobalVal.jobQueue:
	    setJobQueue(Queue.Queue(maxsize = 100))
    return GlobalVal.jobQueue.get(block)

# job queue ops
def setRetQueue(retQueue):
    GlobalVal.retQueue = retQueue

def getRetQueue():
    return GlobalVal.retQueue

def putRet(item, block = True):
    if None == GlobalVal.retQueue:
        setRetQueue(Queue.Queue(maxsize = 100))
    GlobalVal.retQueue.put(item, block)

def getRet(block = True):
    if None == GlobalVal.retQueue:
        setRetQueue(Queue.Queue(maxsize = 100))
    return GlobalVal.retQueue.get(block)
