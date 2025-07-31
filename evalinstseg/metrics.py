import logging
import os
import toml

logger = logging.getLogger(__name__)


class Metrics:
    """class that stores variety of computed metrics

    Attributes
    ----------
    metricsDict: dict
        python dict containing results.
        filled by calling add/getTable and addMetric.
    fn: str/Path
        filename without toml extension. results will be written to this file.
    """
    def __init__(self, fn):
        self.metricsDict = {}
        self.fn = fn

    def save(self):
        """dump results to toml file."""
        logger.info("saving %s", self.fn)
        with open(self.fn+".toml", 'w') as tomlFl:
            toml.dump(self.metricsDict, tomlFl)

    def addTable(self, name, dct=None):
        """add new sub-table to result dict

        pass name containing '.' for nested tables,
        e.g., passing "confusion_matrix.th_0_5" results in:
        `dict = {"confusion_matrix": {"th_0_5": result}}`
        """
        levels = name.split(".")
        if dct is None:
            dct = self.metricsDict
        if levels[0] not in dct:
            dct[levels[0]] = {}
        if len(levels) > 1:
            name = ".".join(levels[1:])
            self.addTable(name, dct[levels[0]])

    def getTable(self, name, dct=None):
        """access existing sub-table in result dict

        pass name containing '.' to access nested tables.
        """
        levels = name.split(".")
        if dct is None:
            dct = self.metricsDict
        if len(levels) == 1:
            return dct[levels[0]]
        else:
            name = ".".join(levels[1:])
            return self.getTable(name, dct=dct[levels[0]])

    def addMetric(self, table_name, name, value):
        """add result for metric `name` to sub-table `table_name` """
        tbl = self.getTable(table_name)
        tbl[name] = value

