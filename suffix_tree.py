import sys
class Node(object):
    """A node in the suffix tree.

    suffix_node
        the index of a node with a matching suffix, representing a suffix link.
        -1 indicates this node has no suffix link.
    occurences
         a 2-tuple containing the document number and the indices that substring occurs
    """
    def __init__(self):
        self.suffix_node = -1
        #self.indices=[]

    def __repr__(self):
        return "Node(suffix link: %d)"%self.suffix_node

class Edge(object):
    """An edge in the suffix tree.

    first_char_index
        index of start of string part represented by this edge

    last_char_index
        index of end of string part represented by this edge

    source_node_index
        index of source node of edge

    dest_node_index
        index of destination node of edge
    """
    def __init__(self, first_char_index, last_char_index, source_node_index, dest_node_index):
        self.first_char_index = first_char_index
        self.last_char_index = last_char_index
        self.source_node_index = source_node_index
        self.dest_node_index = dest_node_index

    @property
    def length(self):
        return self.last_char_index - self.first_char_index

    def __repr__(self):
        return 'Edge(%d, %d, %d, %d)'% (self.source_node_index, self.dest_node_index
                                        ,self.first_char_index, self.last_char_index )


class Suffix(object):
    """Represents a suffix from first_char_index to last_char_index.
    this class rerpesents suffix links
    source_node_index
        index of node where this suffix starts

    first_char_index
        index of start of suffix in string

    last_char_index
        index of end of suffix in string
    """
    def __init__(self, source_node_index, first_char_index, last_char_index):
        self.source_node_index = source_node_index
        self.first_char_index = first_char_index
        self.last_char_index = last_char_index

    @property
    def length(self):
        """String depth of the sufix
        """
        return self.last_char_index - self.first_char_index

    def explicit(self):
        """A suffix is explicit(returns true) if it ends on a node. first_char_index
        is set greater than last_char_index to indicate this.
        """
        return self.first_char_index > self.last_char_index

    def implicit(self):
        return self.last_char_index >= self.first_char_index






class SuffixTree(object):
    """A suffix tree for string matching. Uses Ukkonen's algorithm
    for construction.
    """
    def __init__(self, string):
        self.string = string.lower()
        self.N = len(string) - 1
        self.nodes = [Node()]
        self.edges = {} # Key: value:: (source_node_index,first character of the edge): edge object
        self.active = Suffix(0, 0, -1) # initialise the first active point to be the root


        #builing the tree in phases
        for i in range(len(string)):
            self._add_prefix(i)

    def __repr__(self):
        """
        Lists edges in the suffix tree
        """
        curr_index = self.N
        s = "\tStart \tEnd \tSuf \tFirst \tLast \tString\n"
        values = list(self.edges.values())
        values.sort(key=lambda x: x.source_node_index)
        for edge in values:
            if edge.source_node_index == -1:
                continue
            s += "\t%s \t%s \t%s \t%s \t%s \t"%(edge.source_node_index
                    ,edge.dest_node_index
                    ,self.nodes[edge.dest_node_index].suffix_node
                    ,edge.first_char_index
                    ,edge.last_char_index)


            top = min(curr_index, edge.last_char_index)
            s += self.string[edge.first_char_index:edge.first_char_index+10] + "\n"
            #s += self.string[edge.first_char_index:top+1] + "\n"
        return s

    def _add_prefix(self, last_char_index):
        """The core construction method.
            Insert all suffixes from string[i] to sting[last_char_index]
            called N times
        """
        last_parent_node = -1 # initialisation: root has no parent
        while True:
            parent_node = self.active.source_node_index
            if self.active.explicit():
                if (self.active.source_node_index, self.string[last_char_index]) in self.edges:
                    # prefix is already in tree ( Rule 3)
                    e=self.edges[self.active.source_node_index, self.string[last_char_index]]
                    #self.nodes[self.active.source_node_index].indices.append((0,last_char_index-e.length))
                    break
            else:
                e = self.edges[self.active.source_node_index, self.string[self.active.first_char_index]]
                if self.string[e.first_char_index + self.active.length + 1] == self.string[last_char_index]:
                    # prefix is already in tree (Rule 3)
                    break
                parent_node = self._split_edge(e, self.active) #Rule 2

            self.nodes.append(Node())
            e = Edge(last_char_index, self.N, parent_node, len(self.nodes) - 1)
            self._insert_edge(e)

            if last_parent_node > 0:
                self.nodes[last_parent_node].suffix_node = parent_node
            last_parent_node = parent_node

            if self.active.source_node_index == 0:
                self.active.first_char_index += 1
            else:
                self.active.source_node_index = self.nodes[self.active.source_node_index].suffix_node
            self._canonize_suffix(self.active)
        if last_parent_node > 0:
            self.nodes[last_parent_node].suffix_node = parent_node
        self.active.last_char_index += 1
        self._canonize_suffix(self.active)

    def _insert_edge(self, edge):
        self.edges[(edge.source_node_index, self.string[edge.first_char_index])] = edge

    def _remove_edge(self, edge):
        self.edges.pop((edge.source_node_index, self.string[edge.first_char_index]))

    def _split_edge(self, edge, suffix):
        self.nodes.append(Node())
        e = Edge(edge.first_char_index, edge.first_char_index + suffix.length, suffix.source_node_index, len(self.nodes) - 1)
        self._remove_edge(edge)
        self._insert_edge(e)
        self.nodes[e.dest_node_index].suffix_node = suffix.source_node_index  ### need to add node for each edge
        edge.first_char_index += suffix.length + 1
        edge.source_node_index = e.dest_node_index
        self._insert_edge(edge)
        return e.dest_node_index

    def _canonize_suffix(self, suffix):
        """This canonizes the suffix, walking along its suffix string until it
        is explicit or there are no more matched nodes.
        """
        if not suffix.explicit():
            e = self.edges[suffix.source_node_index, self.string[suffix.first_char_index]]
            if e.length <= suffix.length:
                suffix.first_char_index += e.length + 1
                suffix.source_node_index = e.dest_node_index
                self._canonize_suffix(suffix)

    def _find_substring(self, substring):
        """Returns the edge where the substring was found  in string or -1 if it
        is not found.
        """
        if not substring:
            return -1
        #print substring
        substring = substring.lower()
        curr_node = 0
        i = 0
        prev_edge=0
        while i < len(substring):
            edge = self.edges.get((curr_node, substring[i]))
            #print edge
            if not edge:
                if i==0: # no chaacters match
                    print(" no substrings of the string match the given pattern")
                    return 0, -1
                else: # at least one character matched
                    print" the substring was not found but",i,"characters match."
                    return (prev_edge, i)
            ln = min(edge.length + 1, len(substring) - i)
            if substring[i:i + ln] != self.string[edge.first_char_index:edge.first_char_index + ln]:
                j=i
                while(substring[j]==self.string[edge.first_char_index+(j-i)]):
                    j+=1
                print" the substring was not found but",j,"characters match. Start index: "
                return (edge, j)
            i += edge.length + 1
            curr_node = edge.dest_node_index
            prev_edge=edge
        #indices=find_children(curr_node)
        #print edge.first_char_index - len(substring) + ln
        return edge, len(substring)
        #return edge.first_char_index - len(substring) + ln


    # Public methods
    def find_relevant(self,docs, string):
        stories=[self.string[doc.start:doc.end] for doc in docs]
        titles=[doc.title for doc in docs]
        docstarts=[doc.start for doc in docs]
        count=0


        for doc in docs:
            count+=1
        stories.append(string)

        gst=GST(stories)
        relevance=[]
        rank=[]

        for i in range(count):

            substr,start,end=gst.lcs([i,count])
            rank.append([i,titles[i], len(substr), substr])
            relevance.append([start, end-start,i])
        relevance.sort(key=lambda x:(x[1]))
        rank.sort(key = lambda x: x[2])
        rank.reverse()

        relevance.reverse()

        return rank
    def all_occurences(self, string):
        edge,length =self._find_substring(string)
        if(length==-1):
            return -1
        values = list(self.edges.values())
        values.sort(key=lambda x: x.first_char_index)
        occurences=[]
        for e in values:
            if e.source_node_index==edge.dest_node_index and e.first_char_index-(length+1) not in occurences:
                occurences.append(e.first_char_index-(length+1))
            if e==edge and e.first_char_index-(length+1) not in occurences:
                occurences.append(e.first_char_index-(length-1))
        occurences.pop(0)
        return occurences
    def find(self,string):
        edge,length= self._find_substring(string)
        return edge.first_char_index-1

    def has_substring(self, substring):
        return self.find_substring(substring) != -1




class GST():
    """Class representing the suffix tree."""
    def __init__(self, input=''):
        self.root = _SNode()
        self.root.depth = 0
        self.root.idx = 0
        self.root.parent = self.root
        self.root._add_suffix_link(self.root)

        if not input == '':
           self.build(input)

    def _check_input(self, input):
        if isinstance(input, str):
            return 'st'
        elif isinstance(input, list):
            if all(isinstance(item, str) for item in input):
                return 'gst'

    def build(self, x):
        """Builds the Generalised Suffix tree on the given input.
          in turn calls the normal build function affter concatenating
        :param x: List of Strings
        """
        type = self._check_input(x)

        if type == 'st':
            x += next(self._terminalSymbolsGenerator())
            self._build(x)
        if type == 'gst':
            self._build_generalized(x)

    def _build(self, x):
        """Builds a Suffix tree."""
        self.word = x
        u = self.root
        d = 0
        for i in range(len(x)):
            while u.depth == d and u._has_transition(x[d+i]):
                u = u._get_transition_link(x[d+i])
                d = d + 1
                while d < u.depth and x[u.idx + d] == x[i + d]:
                    d = d + 1
            if d < u.depth:
                u = self._create_node(x, u, d)
            self._create_leaf(x, i, u, d)
            if not u._get_suffix_link():
                self._compute_slink(x, u)
            u = u._get_suffix_link()
            d = d - 1
            if d < 0:
                d = 0

    def _create_node(self, x, u, d):
        i = u.idx
        p = u.parent
        v = _SNode(idx=i, depth=d)
        v._add_transition_link(u,x[i+d])
        u.parent = v
        p._add_transition_link(v, x[i+p.depth])
        v.parent = p
        return v

    def _create_leaf(self, x, i, u, d):
        w = _SNode()
        w.idx = i
        w.depth = len(x) - i
        u._add_transition_link(w, x[i + d])
        w.parent = u
        return w

    def _compute_slink(self, x, u):
        d = u.depth
        v = u.parent._get_suffix_link()
        while v.depth < d - 1:
            v = v._get_transition_link(x[u.idx + v.depth + 1])
        if v.depth > d - 1:
            v = self._create_node(x, v, d-1)
        u._add_suffix_link(v)


    def _build_generalized(self, xs):
        """Builds a Generalized Suffix Tree (GST) from the array of strings provided.
        """
        terminal_gen = self._terminalSymbolsGenerator()

        _xs = ''.join([x + next(terminal_gen) for x in xs])
        self.word = _xs
        self._generalized_word_starts(xs)
        self._build(_xs)
        self.root._traverse(self._label_generalized)

    def _label_generalized(self, node):
        """Helper method that labels the nodes of GST with indexes of strings
        found in their descendants.
        """
        if node.is_leaf():
            x = {self._get_word_start_index(node.idx)}
        else:
            x = {n for ns in node.transition_links for n in ns[0].generalized_idxs}
        node.generalized_idxs = x

    def _get_word_start_index(self, idx):
        """Helper method that returns the index of the string based on node's
        starting index"""
        i = 0
        for _idx in self.word_starts[1:]:
            if idx < _idx:
                return i
            else:
                i+=1
        return i

    def lcs(self, stringIdxs=-1):
        """Returns the Largest Common Substring of Strings provided in stringIdxs.
        If stringIdxs is not provided, the LCS of all strings is returned.

        ::param stringIdxs: Optional: List of indexes of strings.
        """
        if stringIdxs == -1 or not isinstance(stringIdxs, list):
            stringIdxs = set(range(len(self.word_starts)))
        else:
            stringIdxs = set(stringIdxs)

        deepestNode = self._find_lcs(self.root, stringIdxs)
        start = deepestNode.idx
        end = deepestNode.idx + deepestNode.depth
        return self.word[start:end], start, end

    def _find_lcs(self, node, stringIdxs):
        """Helper method that finds LCS by traversing the labeled GSD."""
        nodes = [self._find_lcs(n, stringIdxs)
            for (n,_) in node.transition_links
            if n.generalized_idxs.issuperset(stringIdxs)]

        if nodes == []:
            return node

        deepestNode = max(nodes, key=lambda n: n.depth)
        return deepestNode

    def _generalized_word_starts(self, xs):
        """Helper method returns the starting indexes of strings in GST"""
        self.word_starts = []
        i = 0
        for n in range(len(xs)):
            self.word_starts.append(i)
            i += len(xs[n]) + 1
    def _edgeLabel(self, node, parent):
        """Helper method, returns the edge label between a node and it's parent"""
        return self.word[node.idx + parent.depth : node.idx + node.depth]


    def _terminalSymbolsGenerator(self):
        """Generator of unique terminal symbols used for building the Generalized Suffix Tree.
        Unicode Private Use Area U+E000..U+F8FF is used to ensure that terminal symbols
        are not part of the input string.
        """
        py2 = sys.version[0] < '3'
        UPPAs = list(list(range(0xE000,0xF8FF+1)) + list(range(0xF0000,0xFFFFD+1)) + list(range(0x100000, 0x10FFFD+1)))
        for i in UPPAs:
            if py2:
                yield(unichr(i))
            else:
                yield(chr(i))

class _SNode():
    """Class representing a Node in the Suffix tree."""
    def __init__(self, idx=-1, parentNode=None, depth=-1):
        # Links
        self._suffix_link = None
        self.transition_links = []
        # Properties
        self.idx = idx
        self.depth = depth
        self.parent = parentNode
        self.generalized_idxs = {}

    def __str__(self):
        return("SNode: idx:"+ str(self.idx) + " depth:"+str(self.depth) +
            " transitons:" + str(self.transition_links))

    def _add_suffix_link(self, snode):
        self._suffix_link = snode

    def _get_suffix_link(self):
        if self._suffix_link != None:
            return self._suffix_link
        else:
            return False

    def _get_transition_link(self, suffix):
        for node,_suffix in self.transition_links:
            if _suffix == '__@__' or suffix == _suffix:
                return node
        return False

    def _add_transition_link(self, snode, suffix=''):
        tl = self._get_transition_link(suffix)
        if tl: # TODO: improve this.
            self.transition_links.remove((tl,suffix))
        self.transition_links.append((snode,suffix))

    def _has_transition(self, suffix):
        for node,_suffix in self.transition_links:
            if _suffix == '__@__' or suffix == _suffix:
                return True
        return False

    def is_leaf(self):
        return self.transition_links == []

    def _traverse(self, f):
        for (node,_) in self.transition_links:
            node._traverse(f)
        f(self)

    def _get_leaves(self):
        if self.is_leaf():
            return [self]
        else:
            return [x for (n,_) in self.transition_links for x in n._get_leaves()]
