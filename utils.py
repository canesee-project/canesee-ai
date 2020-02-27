import json
from typing import *
import datetime


def pretty_objects(ugly_objs: List[Tuple[AnyStr, AnyStr]]) -> List[Dict[AnyStr, AnyStr]]:
    """
    Converts a list of tuples to a list of objects.
    """
    return list(map(lambda t: {'label': t[0], 'pos': t[1]}, ugly_objs))


def json_of_result(res_type: int, res_value: Any) -> AnyStr:
    return json.dumps({'type': res_type, 'value': res_value})


def log(*msgs: AnyStr):
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S: "), *msgs)
