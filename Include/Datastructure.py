# arman simi
"""
+Array
+Queue
+Dynamic Hash Table
+Sll
+Close Hash Table
+Trie : Implement With Close Hash Table
+BST
+Chaining Hash Table(Open Hash Table)
"""
from typing import TypeVar, Generic, List, Union
import ctypes
from time import perf_counter, process_time

start_perf = perf_counter()
start_process = process_time()

KeyType = TypeVar("KeyType")
ValueType = TypeVar("ValueType")


class Array:
    def __init__(self, size):
        self.array = (ctypes.py_object * size)()
        for i in range(size):
            self.array[i] = None
        self.len = 0
        self.size = size

    def insert(self, value: int):
        key = self.len
        if self.len == self.size:
            return
        if self.array[key] is None:
            self.array[key] = value
            self.len += 1
            return
        temp = self.size - 1
        if temp is None:
            while temp != key:
                self.array[temp] = self.array[temp-1]
                temp -= 1
            self.array[temp] = value
            self.len += 1

    def delete(self, key: int):
        if self.len == 0:
            return
        del_value = self.array[key]
        if key == (self.size - 1):
            self.array[key] = None
        else:
            while key != self.size:
                if key == (self.size - 1):
                    self.array[key] = None
                else:
                    self.array[key] = self.array[key+1]
                key += 1
        self.len -= 1
        return del_value

    def insert_sorted(self, value: int):
        if self.len == self.size:
            return
        if self.len == 0:
            self.array[0] = value
            self.len = 1
        else:
            i = 0
            while self.array[i] < value:
                if self.array[i+1] is not None:
                    i += 1
                else:
                    break
            if self.array[i] > value:
                temp = self.len
                while temp != i:
                    self.array[temp] = self.array[temp - 1]
                    temp -= 1
                self.array[temp] = value
            elif self.array[i + 1] is None:
                self.array[i + 1] = value
            self.len += 1

    def get_array(self):
        return self.array


class Queue(Generic[ValueType]):
    def __init__(self, size: int):
        self.queue = [None for _ in range(size)]
        self.front = -1
        self.rear = -1

    def size(self):
        return (len(self.queue) - self.front + self.rear) % len(self.queue)

    def enqueue(self, value: ValueType):
        if (self.rear + 1) % len(self.queue) == self.front:
            return
        elif self.front == -1:
            self.front = 0
            self.rear = 0
            self.queue[self.rear] = value
            return
        self.rear = (self.rear + 1) % len(self.queue)
        self.queue[self.rear] = value

    def dequeue(self):
        if self.front == -1:
            return
        else:
            if self.rear == self.front:
                del_val = self.queue[self.front]
                self.queue[self.front] = None
                self.rear = self.front = -1
                return del_val
            del_val = self.queue[self.front]
            self.queue[self.front] = None
            self.front = (self.front + 1) % len(self.queue)
            return del_val

    def show_first(self):
        if self.front == -1:
            return None
        return self.queue[self.front]

    def show_next_first(self):
        if self.front == -1:
            return None
        temp = self.front
        temp = ((temp + 1) % len(self.queue))
        return self.queue[temp]

    def show_last(self):
        if self.front == -1:
            return None
        return self.queue[self.rear]


class DynamicTable(Generic[KeyType, ValueType]):

    def __init__(self, size: int = 5):
        self.len = 0
        self.table: List[Union[None, DynamicTable.DNode]] = [None for _ in range(size)]
        self.size = size
        self.lf = self.len / self.size
        self.low_threshold = 1/4
        self.height_threshold = 3/4
        self.new_table = None

    class DNode(Generic[KeyType, ValueType]):
        def __init__(self, key: KeyType, value: ValueType):
            self.key: KeyType = key
            self.value: ValueType = value

        def __repr__(self):
            return f"node[{self.key},{self.value}]"

        def __iter__(self):
            return self

        def __next__(self):
            return self.key, self.value

    def __repr__(self):
        return repr(self.table)

    def __iter__(self):
        return DynamicTable.DNode

    def get_len(self):
        return self.len

    def hash_func(self, key: KeyType, n: int) -> int:
        hash_result = hash(key)
        m = self.size
        return ((hash_result % m) + n) % m

    def compact(self):
        self.size = round(self.size / 2)
        self.new_table = [None for _ in range(self.size)]

    def expand(self):
        self.size *= 2
        self.new_table = [None for _ in range(self.size)]

    def locate(self, size):
        for i in range(size):
            if self.table[i] is None:
                continue
            self.insert_new_table(self.table[i].key, self.table[i].value, 0)

    def insert_new_table(self, key, value, n: int = 0):
        index = self.hash_func(key, n)
        if self.new_table[index] is None or self.new_table[index].value == "deleted":
            node = DynamicTable.DNode(key, value)
            self.new_table[index] = node
            return
        elif self.new_table[index].key != key:
            self.insert_new_table(key, value, n+1)

    def insert(self, key: KeyType, value: ValueType, i: int = 0):
        index = self.hash_func(key, i)
        if self.len >= self.height_threshold * self.size:
            size = self.size
            self.expand()
            self.locate(size)
            self.table = self.new_table
            self.new_table = None
            self.lf = self.len / self.size
            self.insert(key, value, i)
        else:
            if self.table[index] is None or self.table[index].value == "deleted":
                node = DynamicTable.DNode(key, value)
                self.table[index] = node
                self.len += 1
                self.lf = self.len / self.size
                return
            if self.table[index].key == key:
                self.table[index].key = key
                self.table[index].value = value
                return
            if self.table[index].key != key:
                self.insert(key, value, i+1)

    def delete(self, key: KeyType):
        if self.len <= self.low_threshold * self.size:
            size = self.size
            self.compact()
            self.locate(size)
            self.table = self.new_table
            self.new_table = None
            self.lf = self.len / self.size
            self.delete(key)
        else:
            i = 0
            while True:
                index = self.hash_func(key, i)
                if self.table[index] is None:
                    return
                else:
                    if self.table[index].key == key:
                        self.table[index].value = "deleted"
                        self.len -= 1
                        self.lf = self.len / self.size
                        return
                    elif self.table[index].key != key:
                        i += 1

    def get_value(self, key: KeyType):
        i = 0
        while True:
            index = self.hash_func(key, i)
            if isinstance(self.table[index], DynamicTable.DNode):
                if self.table[index].key == key:
                    if self.table[index].value == "deleted":
                        return None
                    return self.table[index].value
                else:  # elif self.table[index].key != key
                    i += 1
            else:
                return None

    def find(self, key: KeyType):
        i = 0
        while True:
            index = self.hash_func(key, i)
            if isinstance(self.table[index], DynamicTable.DNode):
                if self.table[index].key == key:
                    if self.table[index].value == "deleted":
                        return False
                    return True
                else:
                    i += 1
            else:
                return False


class Sll(Generic[ValueType]):
    class SllNode(Generic[ValueType]):
        def __init__(self, data: ValueType, _next=None):
            self.data = data
            self.next = _next

    def __init__(self):
        self.start = None
        self.last = None
        self.len = 0

    def insert_last(self, value: ValueType):
        node = Sll.SllNode(value)
        if self.start is None:
            self.start = node
            self.last = node
            self.len = 1
            return
        self.last.next = node
        self.last = node
        self.len += 1

    def delete_first(self):
        if self.len == 0:
            return
        if self.len == 1:
            self.start = None
            self.last = None
            self.len = 0
            return
        temp = self.start
        self.start = self.start.next
        temp.next = None
        del temp

    def traverse(self):
        if self.start is None:
            return
        temp = self.start
        while temp:
            yield temp.data
            temp = temp.next

    def get_len(self):
        return self.len


class CloseHash(Generic[KeyType, ValueType]):
    class Node(Generic[KeyType, ValueType]):
        def __init__(self, key: KeyType, value: ValueType):
            self.key: KeyType = key
            self.value: ValueType = value

        def __repr__(self):
            return f"node[{self.key},{self.value}]"

    def __init__(self, initialize_size: int = 13):
        self.table: List[Union[None, CloseHash.Node, str]] = [None for _ in range(initialize_size)]
        self.len = 0

    def __repr__(self):
        return repr(self.table)

    def hash_func(self, key: KeyType, x):
        result = hash(key)
        return ((result % len(self.table)) + x) % 7

    def insert(self, key: KeyType, value: ValueType = None):
        i = 0
        while True:
            index = self.hash_func(key, i)
            if self.table[index] is None or self.table[index] == "deleted":
                self.table[index] = CloseHash.Node(key, value)
                self.len += 1
                return
            if self.table[index].key == key:
                return
            if self.table[index].key != key:
                i += 1

    def delete(self, key: KeyType):
        i = 0
        while True:
            index = self.hash_func(key, i)
            if self.table[index] is None:
                return
            else:
                if isinstance(self.table[index], CloseHash.Node):
                    if self.table[index].key == key:
                        self.table[index] = "deleted"
                        self.len -= 1
                        return
                    else:
                        i += 1
                else:  # label deleted
                    i += 1

    def find(self, key: KeyType):
        i = 0
        while True:
            index = self.hash_func(key, i)
            if isinstance(self.table[index], CloseHash.Node):
                if self.table[index].key == key:
                    return True, index
                i += 1  # self.table[index].key != key
            else:
                if self.table[index] == "deleted" or self.table[index] is None:
                    return False, -1


class Trie(Generic[ValueType]):
    class Node:
        def __init__(self):
            self.mark = False
            self.edges = CloseHash(37)
            self.value: ValueType = None

    class Edge:
        def __init__(self, start, last, label):
            self.start = start
            self.last = last
            self.label = label

    def __init__(self):
        self.root = Trie.Node()
        self.i = 0  # check for complete delete

    def t_add(self, string: str, value: ValueType):
        return self._add(string, value, self.root, 0)

    def _add(self, string: str, value: ValueType, node, count: int = 0):
        if len(string) == count:
            node.mark = True
            node.value = value
            return node.value
        result = node.edges.find(string[count])
        if result[0]:
            return self._add(string, value, node.edges.table[result[1]].value.last, count + 1)
        else:
            new_node = Trie.Node()
            node.edges.insert(string[count], Trie.Edge(node, new_node, string[count]))
            return self._add(string, value, new_node, count + 1)

    def t_find(self, string: str):
        return self._find(string, self.root, 0)

    def _find(self, string: str, node, count: int = 0):
        if len(string) == count:
            return node.mark
        else:
            result = node.edges.find(string[count])
            if result[0]:
                return self._find(string, node.edges.table[result[1]].value.last, count+1)
            else:
                return False

    def t_get(self, string: str):
        return self._get(string, self.root, 0)

    def _get(self, string: str, node, count: int = 0):
        if len(string) == count:
            return node.value
        else:
            result = node.edges.find(string[count])
            if result[0]:
                return self._get(string, node.edges.table[result[1]].value.last, count+1)
            else:
                return None

    def t_delete(self, string: str):
        self._delete(string, self.root, 0)

    def _delete(self, string: str, node, count: int = 0):
        if len(string) == count:
            node.mark = False
            node.value = None
            if self.i == len(string):
                if node.edges.len == 0:
                    self.i = 0
                    self._complete_delete(string, self.root, 0)
            return
        result = node.edges.find(string[count])
        if result[0]:
            if node.edges.len == 1:
                self.i += 1
            self._delete(string, node.edges.table[result[1]].value.last, count+1)
        else:
            return

    def _complete_delete(self, string: str, node, j: int = 0):
        if len(string) == j:
            return
        result = node.edges.find(string[j])
        temp = node.edges.table[result[1]].value.last
        node.edges.delete(node.edges.table[result[1]].key)
        self._complete_delete(string, temp, j+1)


class BST(Generic[ValueType]):
    class Node(Generic[ValueType]):
        def __init__(self, key: int, value: ValueType):
            self.key: int = key
            self.value: ValueType = value
            self.l_ch = None
            self.r_ch = None

    def __init__(self):
        self.root = None
        self.len = 0

    def insert(self, key: int, data: ValueType):
        if self.root is None:
            self.root = BST.Node(key, data)
            self.len = 1
            return
        parent = None
        temp = self.root
        while temp:
            parent = temp
            if temp.key > key:
                temp = temp.l_ch
            elif temp.key < key:
                temp = temp.r_ch
            else:
                return
        bst_node = BST.Node(key, data)
        if parent.key > key:
            parent.l_ch = bst_node
        else:
            parent.r_ch = bst_node
        self.len += 1

    def search(self, key: int):
        if self.root.key == key:
            return self.root.value
        temp = self.root
        while temp:
            if key < temp.key:
                temp = temp.l_ch
            elif key > temp.key:
                temp = temp.r_ch
            else:
                return temp.value
        return False

    def delete(self, key: int):
        if self.root is None:
            return
        if self.root.key == key:
            if self.root.l_ch and self.root.r_ch:
                del_node = self.root
                help_del_node = self.root.l_ch
                if help_del_node.r_ch is None:
                    help_del_node.r_ch = del_node.r_ch
                    del_node.r_ch = None
                    self.root = help_del_node
                    del del_node
                    self.len -= 1
                    return
                parent = None
                while help_del_node.r_ch:
                    parent = help_del_node
                    help_del_node = help_del_node.r_ch
                help_del_node.key, del_node.key = del_node.key, help_del_node.key
                help_del_node.value, del_node.value = del_node.value, help_del_node.value
                if help_del_node.l_ch:
                    parent.r_ch = help_del_node.l_ch
                    help_del_node.l_ch = None
                else:
                    parent.r_ch = None
                del help_del_node
                self.len -= 1
            elif self.root.l_ch:
                temp = self.root
                self.root = self.root.l_ch
                temp.l_ch = None
                del temp
                self.len -= 1
                return
            elif self.root.r_ch:
                temp = self.root
                self.root = self.root.r_ch
                temp.r_ch = None
                del temp
                self.len -= 1
                return
            else:
                self.root = None
                self.len = 0
                return
        else:
            d_node = self.root
            parent = None
            while d_node.key != key:
                parent = d_node
                if key < d_node.key:
                    d_node = d_node.l_ch
                elif key > d_node.key:
                    d_node = d_node.r_ch
            if d_node is None:
                return
            if d_node.key == key:
                if d_node.l_ch and d_node.r_ch:
                    temp_node = d_node.l_ch
                    if temp_node.r_ch:
                        while temp_node.r_ch:
                            temp_node = temp_node.r_ch
                        temp_node.key, d_node.key = d_node.key, temp_node.key
                        temp_node.value, d_node.value = d_node.value, temp_node.value
                        if temp_node.l_ch:
                            if parent.key > temp_node.key:
                                parent.l_ch = temp_node.l_ch
                            else:
                                parent.r_ch = temp_node.l_ch
                        else:
                            if parent.key > temp_node.key:
                                parent.l_ch = None
                            else:
                                parent.r_ch = None
                        del temp_node
                        self.len -= 1
                    else:
                        temp_node.r_ch = d_node.r_ch
                        d_node.r_ch = None
                        if parent.key > d_node.key:
                            parent.l_ch = temp_node
                        else:
                            parent.r_ch = temp_node
                        d_node.l_ch = None
                        del d_node
                        self.len -= 1
                        return
                elif d_node.r_ch:
                    if parent.key > d_node.key:
                        parent.l_ch = d_node.r_ch
                    else:
                        parent.r_ch = d_node.r_ch
                    d_node.r_ch = None
                    self.len -= 1
                    return
                elif d_node.l_ch:
                    if parent.key > d_node.key:
                        parent.l_ch = d_node.l_ch
                    else:
                        parent.r_ch = d_node.l_ch
                    d_node.l_ch = None
                    self.len -= 1
                    return
                else:
                    if parent.key > d_node.key:
                        parent.l_ch = None
                    else:
                        parent.r_ch = None
                    self.len -= 1


class ChainingHashTable(Generic[KeyType, ValueType]):
    class Node(Generic[KeyType, ValueType]):
        def __init__(self, key: KeyType, value: ValueType, next_=None):
            self.key: KeyType = key
            self.value: ValueType = value
            self.next = next_

    class Sll(Generic[KeyType, ValueType]):
        def __init__(self):
            self.len = 0
            self.start = None
            self.last = None

        def insert_first(self, key: KeyType, value: ValueType):
            new_node = ChainingHashTable.Node(key, value)
            if self.start is None:
                self.start = new_node
                self.last = new_node
            else:
                new_node.next = self.start
                self.start = new_node
            self.len += 1

        def delete_first(self):
            if self.start is None:
                return
            if self.len == 1:
                self.start = None
                self.last = None
                self.len = 0
                return
            temp = self.start
            self.start = self.start.next
            temp.next = None
            del temp
            self.len -= 1

        def insert_last(self, key: KeyType, value: ValueType):
            new_node = ChainingHashTable.Node(key, value)
            if self.start is None:
                self.start = new_node
                self.last = new_node
                self.len = 1
                return
            self.last.next = new_node
            self.last = new_node
            self.len += 1

        def delete_last(self):
            if self.start is None:
                return
            if self.len == 1:
                self.start = None
                self.last = None
                self.len = 0
                return
            temp = self.start
            while temp.next.next is not None:
                temp = temp.next
            temp.next = None
            self.last = temp
            self.len -= 1

        def delete(self, key: KeyType):
            if self.start is None:
                return
            if self.start.key == key:
                self.delete_first()
                return
            if self.last.key == key:
                self.delete_last()
                return
            temp = self.start
            back = None
            while temp:
                if temp.key == key:
                    back.next = temp.next
                    temp.next = None
                    self.len -= 1
                back = temp
                temp = temp.next

        def traverse(self):
            if self.start is None:
                return
            temp = self.start
            while temp:
                print(temp.value, end=" ")
                temp = temp.next

        def find(self, key: KeyType):
            if self.start is None:
                return False
            if self.start.key == key:
                return True
            if self.last.key == key:
                return True
            temp = self.start
            while temp:
                if temp.key == key:
                    return True
                temp = temp.next
            return False

    def __init__(self, size: int = 13):
        self.table: List[Union[None, ChainingHashTable.Sll]] = [None for _ in range(size)]

    def hash_func(self, key: KeyType):
        hash_result = hash(key)
        return hash_result % len(self.table)

    def insert(self, key: KeyType, value: ValueType):
        index = self.hash_func(key)
        if self.table[index] is None:
            self.table[index] = ChainingHashTable.Sll()
            self.insert(key, value)
            return
        else:
            self.table[index].insert_first(key, value)
            return

    def delete(self, key: KeyType):
        index = self.hash_func(key)
        if self.table[index] is None:
            return
        if self.table[index].len != 0:
            self.table[index].delete(key)
            if self.table[index].len == 0:
                self.table[index] = None

    def search(self, key: KeyType):
        index = self.hash_func(key)
        if self.table[index] is None:
            return False
        return self.table[index].find(key)


end_perf = perf_counter()
end_process = process_time()

print("perf_counter", end_perf - start_perf)
print("process_counter", end_process - start_process)
