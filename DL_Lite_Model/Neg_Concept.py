from DL_Lite_Model.Concept import Concept
from Utils import S_OMEGA


class Neg_Concept(Concept):
    def __init__(self, concept: Concept):
        self.concept = concept

    def apply_ops(self, paras):
        paras = self.concept.apply_ops(paras)

        data_l_not = -1 * paras['data_l']
        data_l_not[-1] = data_l_not[-1] - S_OMEGA

        data_u_not = -1 * paras['data_u']
        data_u_not[-1] = data_u_not[-1] + S_OMEGA

        return {'col_l': paras['col_l'],
                'data_l': data_l_not,
                'col_u': paras['col_u'],
                'data_u': data_u_not}

    def get_name(self):
        return self.concept.get_name()

    def get_op_name(self):
        return 'neg(' + self.concept.get_op_name() + ')'
