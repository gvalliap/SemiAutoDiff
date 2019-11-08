import numpy as np

def flatten(arr):
  result = arr[0]
  for n in arr[1:]:
    result += n
  return(result)

class Add:
  @staticmethod
  def apply(node):
    return(flatten(node.parents_fwd)+node.weight)

class Mul:
  @staticmethod
  def apply(node):
    return(np.dot(flatten(node.parents_fwd),node.weight))
  