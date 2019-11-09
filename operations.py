import numpy as np

# Reduces a list of parent results
# If only one parent result, just returns it
def flatten(arr):
  result = arr[0]
  for n in arr[1:]:
    result += n
  return(result)

# f(f0,w) = f0+w
class Add:
  @staticmethod
  def apply(node):
    return(flatten(node.parents_fwd)+node.weight)

  @staticmethod
  def dw(node):
    return(np.ones(node.weight.shape))

  @staticmethod
  def df(node):
    return(np.ones(flatten(node.parents_fwd).shape))



class Mul:
  @staticmethod
  def apply(node):
    return(np.dot(flatten(node.parents_fwd),node.weight))

  @staticmethod
  def dw(node):
    return(flatten(node.parents_fwd))

  @staticmethod
  def df(node):
    return(node.weight)



class L1:
  is_scalar = True

  @staticmethod
  def apply(node):
    return(np.linalg.norm((flatten(node.parents_fwd)-node.weight), ord=1))
  
  @staticmethod
  def dw(node):
    return(2*(flatten(node.parents_fwd)-node.weight))

  @staticmethod
  def df(node):
    return(2*(flatten(node.parents_fwd)-node.weight))