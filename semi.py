import numpy as np

class Node:

  def __init__(self, operation, weight=None, parents=None, is_start=False):
    self.operation = operation
    self.parents = parents
    self.weight = weight
    self.is_start = is_start
    self.grad = None
    self.fwd = None
    self.delta = None
    self.parents_fwd = None

  def feed_forward(self):
    if not self.is_start:
      self.parents_fwd = [p.fwd for p in self.parents]
      self.fwd = self.operation.apply(self)
  
  def calc_delta(self):
    if not self.is_start:
      self.delta = self.operation.df(self)


class Model:

  def __init__(self):
    self.nodes = []

  def add(self, n):
    self.nodes.append(n)
  
  def add_all(self, n):
    self.nodes.extend(n)
  
  def fwd(self):
    for n in self.nodes:
      n.feed_forward()
      n.calc_delta()
    return(self.nodes[-1].fwd)
  
  def backprop(self):
    if (hasattr(self.nodes[-1].operation, "is_scalar")):
      running_delta = self.nodes[-1].delta
      for n in reversed(self.nodes[1:-1]):
        if n.weight is not None:
          n.grad = np.dot(n.operation.dw(n).T, running_delta)
        running_delta = np.dot(running_delta, n.delta.T)
    else:
      raise ValueError('Final node is not a scalar')

      