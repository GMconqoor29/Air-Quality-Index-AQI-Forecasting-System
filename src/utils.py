import sys
import os
from datetime import datetime

class DualLogger(object):
    """
    A utility class that duplicates all console print() statements and writes them 
    to a specified text file simultaneously. Supports use as a context manager.
    """
    def __init__(self, filename="output.txt", mode="w"):
        self.terminal = sys.stdout
        self._closed = False
        self.log = open(filename, mode, encoding="utf-8")
        self.write(f"\n{'='*50}\n--- Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n{'='*50}\n")

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def write(self, message):
        self.terminal.write(message)
        if not self._closed:
            self.log.write(message)
            self.flush()

    def flush(self):
        self.terminal.flush()
        if not self._closed:
            self.log.flush()
        
    def close(self):
        if not self._closed:
            self._closed = True
            self.log.close()
            sys.stdout = self.terminal
