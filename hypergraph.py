import numpy as np
import random as rnd
from itertools import combinations
from itertools import permutations 
import bisect
from drive.MyDrive.EpidemicModel.model3.node import Node
from drive.MyDrive.EpidemicModel.model3.edge import Hyperedge
import networkx as nx
import math


class Hypergraph:
    """
    A epidemicmodel is a hypergraph object consisting of persons as nodes and contacts between persons as hyperedges. 
    Each node points to an edgelist which stands from the edges which the node is contained by.
    """
    nodes = []
    edges = []
    
    def __init__(self, _nodelist = None, is_nodeattr = False, _edgelist = None, is_edgeattr = False):
        self.nodes = []
        self.edges = []
        if _nodelist != None:
            for node in _nodelist:
                self.add_node(node, is_nodeattr)
        if _edgelist != None:
            for edge in _edgelist:
                self.add_edge(edge, is_edgeattr)
    
    def add_node(self, raw_node, is_attr = False):
        if is_attr:
            node = Node(raw_node[0],raw_node[1])
        else:
            node = Node(raw_node)
        self.nodes.append(node) 
        return node
    
    def add_edge(self, raw_edge, is_attr = False, active = False):
        s = 0
        if is_attr:
            edge = Hyperedge(raw_edge[0],raw_edge[1])
        else:
            edge = Hyperedge(raw_edge)
        self.edges.append(edge)
        for node in edge.nodelist:
            if node not in self.nodes:
                self.add_node(node)
            node.edgelist.append(edge)
            if active:
                if node.attr_list['state']=='i' or node.attr_list['state']=='a':
                    s += 1
        return s, edge       
    
    def expand_edge(self, edge, node):
        if edge not in self.edges:
            pass
        else:
            edge.nodelist.append(node)
            node.edgelist.append(edge)
    
    def del_node(self,node):
        '''
        Delete a node from the hypergraph.
        Parameter:
        -----
        node_id: int 
        The node's id which node is to be deleted
        '''
        if node not in self.nodes:
            print("The node is not in the hypergraph.")
        else:
            for edge in node.edgelist:
                edge.nodelist.remove(node)
            self.nodes.remove(node)
            del node

    def del_edge(self,edge):
        '''
        Delete a hyperedge from the hypergraph.
        Parameter:
        -----
        edge_id: int 
        The hyperedge's id which hyperedge is to be deleted
        '''
        if edge not in self.edges:
            print("The hyperedge is not in the hypergraph.")
        else:
            for node in edge.nodelist:
                node.edgelist.remove(edge)
            self.edges.remove(edge)
            del edge.nodelist
            del edge.attr_list
            del edge
            
    def set_attr(self, node, attr_list_):
        node.attr_list = attr_list_
    
    def is_adjacent(self, node_1,node_2):
        is_adj = False
        for edge in node_1.edgelist:
             if edge.size() == 2 and (node_2 in edge.nodelist):
                    is_adj = True
                    return is_adj
        return is_adj
    
    def is_hyperedge(self, node_set):
        for edge in self.edges:
            i = 0
            while i < len(node_set) and node_set[i] in edge.nodelist:
                i+=1
            if i == len(node_set):
                return True
        return False


    def edge_to_clique(self, edge):
        '''
        Replace a hyperedge in the hypergraph with 2-uniform edges forming a clique on the nodelist of the hyperedge. 
        '''
        if len(edge.nodelist) > 2:
            for nodepair in combinations(edge.nodelist, 2):
                attr_list = edge.attr_list.copy()
                attr_list['spreading rate'] = attr_list['spreading rate']/(edge.size()-1)
                new_edge = [list(nodepair), attr_list]
                s, new_edge = self.add_edge(new_edge ,True, True)
                new_edge.attr_list['number of infectious'] = s
        else:
            pass
        
    def filter_hyperedges(self, size):
        e = []
        for edge in self.edges:
            if len(edge.nodelist) == size:
                e.append(edge)
        self.edges = e
            
    def to_graph(self):
        for edge in self.edges:
            self.edge_to_clique(edge)
        self.filter_hyperedges(2)
 
    def graph_to_networkx(self):
        G = nx.Graph()
        for edge in self.edges:
            if len(edge.nodelist) > 1:
                G.add_edge(edge.nodelist[0].attr_list['id'], edge.nodelist[1].attr_list['id'])
        return G

    def degree_distribution(self):
        distr=[0]
        for node in self.nodes:
            if len(distr) < node.degree() + 1:
                distr += [0] * (node.degree() + 1-len(distr))
            distr[node.degree()] += 1
        return distr

    def edge_size_distribution(self):
        distr=[0]
        for edge in self.edges:
            if len(distr) < edge.size() + 1:
                distr += [0] * (edge.size() + 1-len(distr))
            distr[edge.size()] += 1
        return distr
    
    def vertex_expansion_of_node_set(self, node_set):
        return len(self.boundary_of_node_set(node_set))/len(node_set)
    
    def boundary_of_node_set(self,node_set):
        delta_v_set = []
        for node in node_set:
            for edge in node.edgelist:
                for node_ in edge.nodelist:
                    if node_ not in delta_v_set and node_ not in node_set:
                        delta_v_set.append(node_)
        return delta_v_set
    
    def vertex_expansion(self):
        minimum = len(self.nodes)
        for i in range(int(len(self.nodes)/2)):
            for node_set in combinations(self.nodes, i+1):
                act_exp = self.vertex_expansion_of_node_set(list(node_set))
                if minimum > act_exp:
                    minimum = act_exp
        return minimum
    
    #def hyperedge_expansion_of_node_set(self, node_set):
        #return len(self.outgoing_edges_of_node_set())/len(node_set)
    
    def adjacency_tensor(self):
        n = len(self.nodes)
        d = len(self.edges[0].nodelist)
        shape = np.ones(d, dtype=int) * n
        tensor = np.zeros(shape)
        for node_set in combinations(self.nodes, d):
            if self.is_hyperedge(node_set):
                indexcomb = [node.attr_list['id'] for node in node_set]
                for index in permutations(indexcomb, d):
                    tensor[index] += 1
        return tensor

    def banerjee_adj_mtx(self):
        n = len(self.nodes)
        shape = np.ones(2, dtype=int) * n
        mtx = np.zeros(shape)
        for edge in self.edges:
            for node_pair in combinations(edge.nodelist,2):
                indexcomb = [node.attr_list['id'] for node in node_pair]
                for index in permutations(indexcomb, 2):
                    mtx[index] += 1/(len(edge.nodelist)-1)
        return mtx

    def degree_diag_mtx(self):
        return np.diag(self.degree_vec())
    

    def banerjee_laplacian_mtx(self):
        ddiag = self.degree_diag_mtx()
        mtx = self.banerjee_adj_mtx()
        return  (ddiag - mtx)
        
    def incidence_mtx(self):
        shape = (len(self.nodes),len(self.edges))
        mtx = np.zeros(shape)
        j = 0
        for edge in self.edges:
            i=0
            for node in self.nodes:
                if edge in node.edgelist:
                    mtx[i][j] = 1
                i+=1
            j+=1
        return mtx

    def degree_vec(self):
        return np.array([node.degree() for node in self.nodes])
    
    def weighted_degree_vec(self):
        return np.array([node.weighted_degree() for node in self.nodes])
        
    def wieghted_edge_size_vec(self):
        return np.array([edge.weighted_size() for edge in self.edges])
        
    def node_weight_vec(self):
        return np.array([node.attr_list['weight'] for node in self.nodes])
    
    def edge_weight_vec(self):
        return np.array([edge.attr_list['weight'] for edge in self.edges])
        
    def spectral_centrality(self,mode, error_limit=0.00001, graph_exp=True):
        n=len(self.nodes)
        m=len(self.edges)
        B=self.incidence_mtx()
        if mode=='clique':
            mtx= np.matmul(B,B.transpose())
            w,v=np.linalg.eig(mtx)
            importance_vec=v[:,np.argmax(w)].real
            return importance_vec
        if mode=='bipartite':
            mtx1=np.concatenate((np.zeros((n,n)),np.matmul(B,np.diag(self.edge_weight_vec()))),axis=1)
            mtx2=np.concatenate((np.matmul(B.transpose(),np.diag(self.node_weight_vec())),np.zeros((m,m))),axis=1)
            mtx=np.concatenate((mtx1,mtx2))
            if graph_exp:
                G = nx.from_numpy_matrix(mtx)
                centrality=nx.eigenvector_centrality(G, max_iter=500, tol=error_limit, nstart=None, weight='weight')
                importance_vec=sorted(centrality.items())
                importance_vec=[e[1] for e in importance_vec]
            else:
                w,v=np.linalg.eig(mtx)
                importance_vec=v[:,np.argmax(w)].real
            return importance_vec[:n], importance_vec[n:n+m]
        if mode=='tensor':
            pass
        if mode=='tudisco':
            W = self.edge_weight_vec()
            N = self.node_weight_vec()
            x2,y2 = self.weighted_degree_vec(),self.wieghted_edge_size_vec()
            x1,y1 = np.ones(len(x2)), np.ones(len(y2)) 
            while (np.linalg.norm(x2-x1)/np.linalg.norm(x2) + np.linalg.norm(y2-y1)/np.linalg.norm(y1))<error_limit:
                x1 = x2
                y1 = y2
                g = element_wise_array_f(np.matmul(np.matmul(B,W),y1),lambda x:1/(x**2))
                u = element_wise_array_f(element_wise_array_multiplication(x1,g),math.sqrt)
                psi = element_wise_array_f(np.matmul(np.matmul(B.transpose(),N),element_wise_array_f(x1,math.log)),math.exp)
                v = element_wise_array_f(element_wise_array_multiplication(y1,psi),math.sqrt)
                x2 = u/np.linalg.norm(u)
                y2 = v/np.linalg.norm(v)
            return x2,y2
    def PageRank(self,alpha=0.85, personalization=None,
                max_iter=200, tol=1.0e-6, nstart=None, weight='weight',
                dangling=None, mode= 'bipartite'):
        n=len(self.nodes)
        m=len(self.edges)
        B=self.incidence_mtx()
        if mode=='bipartite':
            mtx1=np.concatenate((np.zeros((n,n)),np.matmul(B,np.diag(self.edge_weight_vec()))),axis=1)
            mtx2=np.concatenate((np.matmul(B.transpose(),np.diag(self.node_weight_vec())),np.zeros((m,m))),axis=1)
            mtx=np.concatenate((mtx1,mtx2))
            G = nx.from_numpy_matrix(mtx)
            pr=nx.pagerank(G,alpha, personalization,max_iter, tol, nstart, weight, dangling)
            pr_vec = list(pr.values())
        return pr_vec[:n], pr_vec[n:n+m]
    
    def to_bipartite_nxGraph_expansion(self):
        n=len(self.nodes)
        m=len(self.edges)
        B=self.incidence_mtx()
        mtx1=np.concatenate((np.zeros((n,n)),np.matmul(B,np.diag(self.edge_weight_vec()))),axis=1)
        mtx2=np.concatenate((np.matmul(B.transpose(),np.diag(self.node_weight_vec())),np.zeros((m,m))),axis=1)
        mtx=np.concatenate((mtx1,mtx2))
        return nx.from_numpy_matrix(mtx)

def element_wise_array_f(array,f):
  return np.array(list(map(f,array)))
 
def element_wise_array_multiplication(arr1,arr2):
  return np.array([arr1[i]*arr2[i] for i in range(len(arr1))])
  