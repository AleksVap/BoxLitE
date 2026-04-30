from DL_Lite_Model.Role import Role


class Inverse_Role(Role):
    def __init__(self, role: Role):
        self.role = role

    def apply_ops(self, paras):
        paras = self.role.apply_ops(paras)
        return {'col_hl': paras['col_tl'],
                'data_hl': paras['data_tl'],
                'col_hu': paras['col_tu'],
                'data_hu': paras['data_tu'],
                'col_tl': paras['col_hl'],
                'data_tl': paras['data_hl'],
                'col_tu': paras['col_hu'],
                'data_tu': paras['data_hu'],
                'col_bl': paras['col_bl'],
                'data_bl': paras['data_bl'],
                'col_bu': paras['col_bu'],
                'data_bu': paras['data_bu']}

    def get_name(self):
        return self.role.get_name()

    def get_op_name(self):
        return 'inv(' + self.role.get_op_name() + ')'
