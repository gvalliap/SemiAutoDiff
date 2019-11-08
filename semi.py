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
    else:
      pass

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
    return(self.nodes[-1].fwd)