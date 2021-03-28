from collections.abc import Iterable

class linearize:
    
    def __init__(self, sequence):
        self.len = len(sequence)
        self.iter = iter(sequence)
        
    def __iter__(self):
        self.subiter = None
        return self
    
    def __next__(self):
        while (1):
            if self.subiter == None:
                element = next(self.iter)
                if not (isinstance(element, str) and len(element) < 2) and isinstance(element, Iterable):
                    self.subiter = iter(linearize(element))
                else:
                    return element

            # subiter != None

            try:
                element = next(self.subiter)
                return element
            except StopIteration:
                self.subiter = None
