import redis
from datas import Config

redis_conn = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, password=Config.REDIS_PSWD, db=Config.REDIS_DB)


def setValue(key, value):
    print("-----redis :setValue:", str(key), ':', value)
    return redis_conn.set(str(key), value)


def getValue(key):
    value = redis_conn.get(str(key))
    print("-----redis :getValue:", str(key), ':', value)
    return value


def setDict(key, dict):
    print("-----redis :setDict:", str(key), ':', dict)
    return redis_conn.hmset(str(key), dict)


def getDict(key):
    value = redis_conn.hgetall(str(key))
    print("-----redis :getValue:", str(key), ':', value)
    return value


def clearAll():
    redis_conn.flushall()
    print("-----redis :clearAll")
    return True
