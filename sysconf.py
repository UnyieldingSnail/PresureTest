# -*- coding: utf-8 -*-
import configparser
import logging
from logging import config
from functools import wraps
import os


# 单例类
def singleton(cls):
    instances = {}

    @wraps(cls)
    def getinstance(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return getinstance


@singleton
class LogConf:
    def __init__(self):
        # 载入配置信息,从Logging.cfg
        conf_path = os.path.join(os.path.dirname(__file__), 'conf')
        logging_config_path = os.path.join(conf_path, 'logging.conf')
        logging.config.fileConfig(logging_config_path)

    def getLogger(self, name):
        return logging.getLogger(name)


@singleton
class BlockConf:
    def __init__(self):
        # 载入配置信息
        cf = configparser.ConfigParser()
        conf_path = os.path.join(os.path.dirname(__file__), 'conf')
        block_config_path = os.path.join(conf_path, 'block.conf')
        cf.read(block_config_path)
        self.shape = (cf.getint("shape", "row"), cf.getint("shape", "col"))
        self.resize = (cf.getint("resize", "width"), cf.getint("resize", "height"))
        self.rowRatio = [float(x) for x in cf.get("ratio", "row").split(",")]
        self.colRatio = [float(x) for x in cf.get("ratio", "col").split(",")]
