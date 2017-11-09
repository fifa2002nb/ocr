from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import redis, logging
import time

class redisPUBHandler(logging.Handler):
  def __init__(self, topic, host, port, db):
    logging.Handler.__init__(self)
    self.topic = topic
    self.host = host
    self.port = port
    self.db = db
    self.pool = redis.ConnectionPool(host=self.host, port=self.port, db=self.db)  
    self.redis_instance = redis.StrictRedis(connection_pool=self.pool)  
    formatter = logging.Formatter('[%(asctime)s] %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    self.setFormatter(formatter)
    self.setLevel(logging.DEBUG)

  def emit(self, record):
    try:
      msg = self.format(record)
      if None == self.redis_instance:
        self.__init__(self.topic, self.host, self.port, self.db)
      self.redis_instance.publish(self.topic, msg)  
    except e:
      print(e)
      self.pool = None
      self.redis_instance = None
      self.handleError(record)


if __name__ == '__main__':
  pool = redis.ConnectionPool(host='10.10.100.14', port=6379, db=1)  
  r = redis.StrictRedis(connection_pool=pool)  
  p = r.pubsub()  
  p.subscribe('lstm_ctc_ocr')  
  for item in p.listen():      
    if item['type'] == 'message':    
      data = item['data']   
      print data  
    if item['data']=='over':  
      break;  
  p.unsubscribe('lstm_ctc_ocr')  

'''
if __name__ == '__main__':
  redis_logger = redisPUBHandler("lstm_ctc_ocr", "10.10.100.14", 6379, 1)

  logging.basicConfig(
            level       = logging.DEBUG,
            format      = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
            datefmt     = '[%y-%m-%d %H:%M:%S]',
            )
  logging.getLogger('').addHandler(redis_logger)
  logging.debug('[debug] Quick zephyrs blow, vexing daft Jim.')
  logging.info('[info] Jackdaws love my big sphinx of quartz.')
  logging.warning('[warning] Jail zesty vixen who grabbed pay from quack.')
  logging.error('[error] The five boxing wizards jump quickly.')
'''