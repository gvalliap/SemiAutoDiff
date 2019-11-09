import semi as sad
import numpy as np
import operations as op

m1 = sad.Model()
X = sad.Node(None, is_start=True)
X.fwd = np.ones((1,2))
f1 = sad.Node(op.Mul, weight=np.ones((2, 3)), parents=[X])
f2 = sad.Node(op.Mul, weight=np.ones((3, 4)), parents=[f1])
f3 = sad.Node(op.Mul, weight=np.ones((4, 5)), parents=[f2])
f4 = sad.Node(op.L1,  weight=np.random.rand(1, 5), parents=[f3])
m1.add_all([X, f1, f2, f3, f4])
m1.fwd()
m1.backprop()