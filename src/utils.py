class Node():
    __slots__ = 'prev', 'next', 'data'
    def __init__(self, data):
        self.data = data

class LinkedList():
    def __init__(self):
        self.dummy = Node(None)
        self.dummy.prev = self.dummy
        self.dummy.next = self.dummy
        self.size = 0
    
    def remove(self, node):
        if self.size == 0:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1
        
    def append(self, data):
        x = Node(data)
        x.prev = self.dummy.prev
        x.next = self.dummy
        self.dummy.prev.next = x
        self.dummy.prev = x
        self.size += 1
        
    def __len__(self):
        return self.size
