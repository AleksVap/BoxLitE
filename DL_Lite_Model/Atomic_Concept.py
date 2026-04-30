from DL_Lite_Model.Concept import Concept


class Atomic_Concept(Concept):
    def __init__(self, name):
        self.name = name

    def apply_ops(self, paras):
        return paras

    def get_name(self):
        return (self.name, "concept")

    def get_op_name(self):
        return self.get_name()[0]