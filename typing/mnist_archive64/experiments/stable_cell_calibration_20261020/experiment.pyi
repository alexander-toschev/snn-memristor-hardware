from configparser import ConfigParser
INI: str
SIM_INI: str
def preflight(source: str) -> tuple[ConfigParser,dict[str,str]]: ...
