import functools
from typing import List
import numpy as np

from agox.__version__ import __version__

LINE_LENGTH = 79
PADDING_CHARACTER = '='
TERMINATE_CHARACTER = '|'

ICON = f"""
       _            _  _  _        _  _  _  _    _           _ 
     _(_)_       _ (_)(_)(_) _   _(_)(_)(_)(_)_ (_)_       _(_)
   _(_) (_)_    (_)         (_) (_)          (_)  (_)_   _(_)  
 _(_)     (_)_  (_)    _  _  _  (_)          (_)    (_)_(_)    
(_) _  _  _ (_) (_)   (_)(_)(_) (_)          (_)     _(_)_     
(_)(_)(_)(_)(_) (_)         (_) (_)          (_)   _(_) (_)_   
(_)         (_) (_) _  _  _ (_) (_)_  _  _  _(_) _(_)     (_)_ 
(_)         (_)    (_)(_)(_)(_)   (_)(_)(_)(_)  (_)         (_)  v{__version__} \n
"""

def line_breaker(string: str, tab_size: int = 4) -> List[str]:
    """Break a long string into lines to be used in `pretty_print`.

    Parameters
    ----------
    string : str
        The long string to break into lines
    tab_size : int, optional
        The width of a tab character in number of space characters,
        by default 4

    Returns
    -------
    List[str]
        The original string split into individual lines
    """

    string = string.replace('\t', ' ' * tab_size)

    text_length = LINE_LENGTH - 4  # lines are padded with 4 characters
    lines = []
    remainder = string  # every iteration, remainder will be shortened

    while len(remainder) > text_length:
        max_line = remainder[:text_length + 1]  # there might be a space at the end
        space_index = max_line.rfind(' ')
        if space_index != -1:  # space found
            lines.append(remainder[:space_index])
            remainder = remainder[space_index + 1:]  # remove the space
        else:  # no space found, simply break the line at the maximum length
            lines.append(remainder[:text_length])
            remainder = remainder[text_length:]

    lines.append(remainder)  # final piece of the string
    return lines
        
def header_print(string):
    
    string = ' ' + string + ' '
    num_markers = int((LINE_LENGTH - len(string))/2) - 1
    header_string = TERMINATE_CHARACTER + num_markers * PADDING_CHARACTER + string + num_markers * PADDING_CHARACTER + TERMINATE_CHARACTER

    if len(header_string) < LINE_LENGTH:
        header_string = header_string[:-2] + PADDING_CHARACTER*2 + TERMINATE_CHARACTER        

    print(header_string)

def pretty_print(string, *args, **kwargs):
    string = str(string)
    for arg in args:
        string += str(arg)

    all_strings = line_breaker(string)
    for string in all_strings:
        string = TERMINATE_CHARACTER + ' ' + string + (LINE_LENGTH - len(string)-3) * ' ' + TERMINATE_CHARACTER       
        print(string, **kwargs)

def agox_writer(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.use_counter:
            header_print(self.name)
        func(self, *args, **kwargs)
        if self.use_counter and len(self.lines_to_print) > 0:
            header_print(self.name)
            for string, args, kwargs in self.lines_to_print:
                string = self.writer_prefix + str(string)
                pretty_print(string, *args, **kwargs)
        self.lines_to_print = []
        
    return wrapper

# def agox_writer(func):
#     @functools.wraps(func)
#     def wrapper(self, state, *args, **kwargs):
#         if not self.use_counter:
#             header_print(self.name)
#         state = func(self, state, *args, **kwargs)
#         if self.use_counter and len(self.lines_to_print) > 0:
#             header_print(self.name)
#             for string, args, kwargs in self.lines_to_print:
#                 string = self.writer_prefix + str(string)
#                 pretty_print(string, *args, **kwargs)
#         self.lines_to_print = []
        
#     return wrapper

class Writer:

    def __init__(self, verbose=True, use_counter=True, prefix=''):
        self.verbose = verbose
        self.use_counter = use_counter
        self.lines_to_print = []
        self.writer_prefix = prefix

    def writer(self, string, *args, **kwargs):
        if self.verbose:
            if self.use_counter:
                self.lines_to_print.append((string, args, kwargs))
            else:
                string = self.writer_prefix + str(string)
                pretty_print(string, *args, **kwargs)

    def header_print(self, string):
        if self.verbose:
            header_print(string)
