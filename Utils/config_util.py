import re
import yaml
import logging
from types import SimpleNamespace

def auto_cast(val):
    if isinstance(val, str):
        # 匹配科学计数法或小数或整数
        if re.match(r"^-?\d+(\.\d+)?(e-?\d+)?$", val, re.IGNORECASE):
            return float(val)
        return val
    elif isinstance(val, dict):
        return {k: auto_cast(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [auto_cast(v) for v in val]
    return val

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config_dict = auto_cast(config_dict)
    return dict_to_namespace(config_dict)


def log_namespace(namespace, prefix=""):
    """
    递归解析并逐行记录 SimpleNamespace 对象中的所有参数

    参数:
        namespace (SimpleNamespace): 包含参数的对象
        prefix (str): 用于打印嵌套参数的前缀
    """
    for key, value in vars(namespace).items():
        if isinstance(value, SimpleNamespace):  # 如果是嵌套结构
            logging.info(f"{prefix}{key}:")
            log_namespace(value, prefix=prefix + "  ")  # 递归解析
        elif isinstance(value, list):  # 如果是列表
            logging.info(f"{prefix}{key}: {value}")
        else:  # 普通值
            logging.info(f"{prefix}{key}: {value}")