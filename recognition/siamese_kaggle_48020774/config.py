import json

class Config(object):
    _instance = None
    _config_json = None

    def __init__(self):
        raise RuntimeError('Call getInstance() instead')

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            print('Creating new instance')
            cls._instance = cls.__new__(cls)
            cls._config_json = json.load(open('recognition/siamese_kaggle_48020774/utility/config.json'))
            # Put any initialization here.
        return cls._instance

    def __getitem__(self, key: str):
        return self._config_json[key]