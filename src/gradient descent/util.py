import sys
from numpy import *

def raiseNotDefined():
  print("Method not implemented: "+inspect.stack()[1][3])    
  sys.exit(1)

