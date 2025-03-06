import numpy as np
import torch
import math

# Vector object for 3-D Representation of a pose in SE2
class vector():
    def __init__(self, vec_list):
        self._x = vec_list[0]
        self._y = vec_list[1]
        self._theta = vec_list[2]

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def theta(self):
        return self._theta

    @property
    def listView(self):
        # If the item is a tensor, we need to detach it in case we want to cast it to an np item
        if torch.is_tensor(self._x):
            return [self._x.detach(), self._y.detach(), self._theta.detach()]
        return [self._x, self._y, self._theta]

class SE2():
    def __init__(self,
                 vec=None,
                 t_mat=None,
                 man_vec=None,
                 ):
        # Vector representation of a point on an SE2 manifold
        # Should be of the form [x, y, theta]
        if vec is not None:
            self.vec = vector(vec)

        if t_mat is not None:
            x = t_mat[0, 2]
            y = t_mat[1, 2]
            th = np.arctan2(t_mat[1,0], t_mat[0,0])
            self.vec = vector([x, y, th])
        if man_vec is not None:
            if torch.is_tensor(man_vec):
                self.vec = vector([man_vec.detach()[0], man_vec.detach()[1], np.arctan2(man_vec.detach()[2],man_vec.detach()[3])])
                return

            self.vec = vector([man_vec[0], man_vec[1], np.arctan2(man_vec[2],man_vec[3])])


    @property
    def vector(self):
        return self.vec

    # Return the homogenous transformation matrix representation of a given vector
    @property
    def t_matrix(self):
        transformation = torch.tensor([[math.cos(self.vec.theta), -math.sin(self.vec.theta), self.vec.x],
                  [math.sin(self.vec.theta), math.cos(self.vec.theta),  self.vec.y],
                  [0.0                   , 0.0                   ,  1.0       ]])#, requires_grad=True)
        return transformation


def adjoint(xi):
    """Takes adjoint of element in SE(3)

    Args:
        xi (3x3 ndarray) : Element in Lie Group

    Returns:
        Ad_xi (3,3 ndarray) : Adjoint in SE(3)"""
    # make the swap
    xi[0,2], xi[1,2] = xi[1,2], -xi[0,2]
    return xi

def carat(xi):
    """Moves an vector to the Lie Algebra se(3).
    Or X^

    Args:
        xi (3 ndarray) : Parametrization of Lie algebra

    Returns:
        xi^ (3,3 ndarray) : Element in Lie Algebra se(2)"""
    return np.array([[0,   -xi[2], xi[0]],
                    [xi[2], 0,     xi[1]],
                    [0,     0,     0]])


def convertTrajToManifolds(trajectory):
    mans = []
    for pose in trajectory:
        if len(pose.shape) == 1:
            if pose.shape[0] == 4:
                manifold = SE2(man_vec=pose.detach())
            elif pose.shape[0] == 3:
                manifold = SE2(vec=pose)
        else:
            manifold = SE2(t_mat=pose)
        mans.append(manifold)

    return mans

def manifoldToVector(manifold):
    x = manifold[0, 2]
    y = manifold[1, 2]
    th = np.arctan2(manifold[1,0],manifold[0,0])

    return torch.Tensor([x,y,th])


# implementation of the boxplus operation for SE2
def BoxPlus(lhs, rhs):
    x = lhs.vec.x + rhs.vec.x
    y = lhs.vec.y + rhs.vec.y
    th = lhs.vec.theta + rhs.vec.theta
    new_vec = [x, y, th]

    return SE2(new_vec).t_matrix

# implementation of the boxminus operation for SE2
def BoxMinus(lhs, rhs):
    x = (lhs.vec.x - rhs.vec.x).item()
    y = (lhs.vec.y - rhs.vec.y).item()
    th = (vpi(lhs.vec.theta - rhs.vec.theta)).item()
    new_vec = torch.Tensor([x, y, th])

    return new_vec#SE2(new_vec).t_matrix

# implementation of the boxminus operation for SE2
def BoxMinusVector(lhvec, rhvec):
    lhs, rhs = SE2(vec=lhvec), SE2(vec=rhvec)
    x = (lhs.vec.x - rhs.vec.x).item()
    y = (lhs.vec.y - rhs.vec.y).item()
    th = (vpi(lhs.vec.theta - rhs.vec.theta)).item()
    new_vec = torch.Tensor([x, y, th])

    return new_vec

# Vpi implementation based on the description the Vector Space and planar rotations described
# in Integrating generic sensor fusion algorithms with sound state representations
# through encapsulation of manifolds, sec. 3.6.1, 3.6.2
def vpi(delta):
    return delta - 2*math.pi * math.floor(( (delta + math.pi) / (2*math.pi)))
