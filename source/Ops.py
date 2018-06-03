from abc import ABC, abstractmethod

class Ops(ABC):
    def __init__(self, name):
        self.name = name
        pass

    @abstractmethod
    def perform_op(self):
        print ("Performing op:" , self.name)
        pass

class Addition(Ops):

    def __init__(self, name="Addition"):
        Ops.__init__(self, name)

    def perform_op(self):
        Ops.perform_op(self)
        print ("Performed op:"+ self.name)


class Subtraction(Ops):

    def __init__(self, name="Subtraction"):
        Ops.__init__(self, name)

    def perform_op(self):
        print ("Performed op:"+ self.name)


op = Addition()
op.perform_op()
