from abc import abstractmethod


class Role:
    @abstractmethod
    def apply_ops(self, paras):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_op_name(self):
        pass

    def __eq__(self, other):
        return self.get_op_name() == other.get_op_name()

    def __hash__(self):
        return hash(self.get_op_name())