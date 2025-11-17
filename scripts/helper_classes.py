from typing import Any

from lensing_and_precession.modules.functions_ver2 import get_td_from_MLz, get_I_from_y, get_MLz_from_td, get_y_from_I, solar_mass
# from modules.default_params_ver2 import *
# from modules.

class Parameters(dict):

    def __getitem__(self, name: str):
        return super().__getitem__(name)
    
    def __key(self):
        return tuple((k,self[k]) for k in sorted(self))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other) -> bool:
        if type(other) != dict or type(other) != Parameters: return False
        return self.__key() == tuple((k,other[k]) for k in sorted(other))
    

class LensParameters(Parameters):
    def __init__(self, new_params: dict):
        if("y" in new_params and "MLz" in new_params):
            #parameterized in terms of y and MLz, initialize them first
            self.__setitem__("y", new_params["y"])
            self.__setitem__("MLz", new_params["MLz"])
        elif("td" in new_params and "I" in new_params):
            self.__setitem__("I", new_params["I"])
            self.__setitem__("td", new_params["td"])
        super().__init__(new_params)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        if(key == "MLz"):
            super().__setitem__("td", get_td_from_MLz(value / solar_mass, self["y"]))
        elif(key == "y"):
            super().__setitem__("I", get_I_from_y(value))
        elif(key == "td"):
            super().__setitem__("MLz", get_MLz_from_td(value, self["y"]) * solar_mass)
        elif(key == "I"):
            super().__setitem__("y", get_y_from_I(value))
    
    def update(self, *args, **kwargs):
        other = dict(*args, **kwargs)
        for key in other:
            self.__setitem__(key, other[key])