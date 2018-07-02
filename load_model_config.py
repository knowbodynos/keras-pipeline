import os
import yaml
import pprint

class Config(object):
    class Objectify(object):
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    self.__dict__[k] = Config.Objectify(**v)
                elif isinstance(v, (list, tuple)):
                    l = []
                    for vv in v:
                        if isinstance(vv, dict):
                            l.append(Config.Objectify(**vv))
                        else:
                            l.append(vv)
                    self.__dict__[k] = l
                else:
                    self.__dict__[k] = v

        def __setattr__(self, key, value):
            self.__dict__[key] = value

        def __setitem__(self, key, value):
            self.__dict__[key] = value

        def __getattr__(self, key):
            if key in self.__dict__:
                return self.__dict__[key]

        def __getitem__(self, key):
            if key in self.__dict__:
                return self.__dict__[key]

        def __delattr__(self, key):
            if key in self.__dict__:
                del self.__dict__[key]

        def __delitem__(self, key):
            if key in self.__dict__:
                del self.__dict__[key]

        def __iter__(self):
            return iter(self.__dict__.keys())

        def __repr__(self):
            return pprint.pformat(self.__dict__)

        def keys(self):
            return self.__dict__.keys()

        def to_dict(self):
            d = {}
            for k, v in self.__dict__.items():
                if isinstance(v, Config.Objectify):
                    d[k] = v.to_dict()
                elif isinstance(v, (list, tuple)):
                    l = []
                    for vv in v:
                        if isinstance(vv, Config.Objectify):
                            l.append(vv.to_dict())
                        else:
                            l.append(vv)
                    d[k] = l
                else:
                    d[k] = v
            return d

    def __init__(self, config_path = None):
        if not config_path:
            config_path = '.'
        self.__config_path = os.path.abspath(config_path)
        self.reload()

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]

    def __delattr__(self, key):
        if key in self.__dict__:
            del self.__dict__[key]

    def __delitem__(self, key):
        if key in self.__dict__:
            del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__.keys())

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def reload(self):
        with open(self.__config_path + "/model.config", "r") as config_stream:
            model_config = yaml.load(config_stream)
        for k, v in model_config.items():
            if isinstance(v, dict):
                self.__dict__[k] = Config.Objectify(**v)
            elif isinstance(v, (list, tuple)):
                l = []
                for vv in v:
                    if isinstance(vv, dict):
                        l.append(Config.Objectify(**vv))
                    else:
                        l.append(vv)
                self.__dict__[k] = l
            else:
                self.__dict__[k] = v

        if self.scheduler.crit_epochs == None:
            self.scheduler.crit_epochs = []

        if self.scheduler.drop_factors == None:
            self.scheduler.drop_factors = []

        if self.metrics == None:
            self.metrics = []

        for i in range(len(self.hidden)):
            if i == 0:
                assert(not (isinstance(self.hidden[i].act, dict) and "eql" in self.hidden[i].keys()))
            if i < len(self.hidden) - 1:
                assert(self.hidden[i].out_units == self.hidden[i + 1].in_units)
                if isinstance(self.hidden[i].act, dict) and "eql" in self.hidden[i].keys():
                    assert(self.hidden[i].out_units < self.hidden[i].in_units and self.hidden[i].in_units < 2 * self.hidden[i].out_units)
            else:
                assert(not (isinstance(self.hidden[i].act, dict) and "eql" in self.hidden[i].keys()))

        assert(len(self.scheduler.crit_epochs) == len(self.scheduler.drop_factors))