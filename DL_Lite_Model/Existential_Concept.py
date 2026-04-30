from DL_Lite_Model.Concept import Concept
from DL_Lite_Model.Role import Role
import numpy as np


class Existential_Concept(Concept):
    def __init__(self, role: Role):
        self.role = role

    def apply_ops(self, paras):
        paras = self.role.apply_ops(paras)

        return {'col_l': np.concatenate([paras['col_hl'][:-1], paras['col_bu']]),
                'data_l': np.concatenate([paras['data_hl'][:-1], -paras['data_bu']]),
                'col_u': np.concatenate([paras['col_hu'][:-1], paras['col_bl']]),
                'data_u': np.concatenate([paras['data_hu'][:-1], -paras['data_bl']])}

    def get_name(self):
        return self.role.get_name()

    def get_op_name(self):
        return 'exists(' + self.role.get_op_name() + ')'