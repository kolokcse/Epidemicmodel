import numpy as np
import random as rnd
import math
from scipy.stats import poisson
from scipy.stats import pareto
from scipy.stats import gamma
import bisect
from drive.MyDrive.EpidemicModel.model3.node import Node
from drive.MyDrive.EpidemicModel.model3.edge import Hyperedge
from drive.MyDrive.EpidemicModel.model3.hypergraph import Hypergraph


class EpidemicModel(object):
    """
    A epidemicmodel is an object aiming to model an epidemic spreading on a closed population.
    The population is modeled by a hypergraph.
    The hypergraph consists of individuals as nodes and contacts between them as hyperedges. 
    """
    def __init__(self, nodelist, state_transition_probs, infectious_change_mtx = None,
                    states = ['s','e','i','a','r','d'],infectious_states=['i','a'], 
                    number_of_stratas = 1 ,testing_prob = 0,
                    testing_capacity = math.inf, quarantine_length = None, 
                    states_fix_period_time = None, edge_size_limit = math.inf,
                    spreading_function = None, vaccination_method = None,
                    vaccination_limit = None, vaccination_start_time = -1, vaccination_start_d_p=-1;
                    vaccination_efficient = 'total efficient', vaccination_efficiency=1,
                    model = 'basic', exclude_died = False,
                    node_history_on = False, school_closure=False):        
        
        self.state_transition_probs = state_transition_probs
        self.number_of_stratas = number_of_stratas
        for state in states:
            if state not in self.state_transition_probs[state].keys():
                keys=list(self.state_transition_probs[state].keys()).copy()
                self.state_transition_probs[state][state] = []
                for age in range(self.number_of_stratas):
                    s = 0
                    for state_2 in keys:
                        if type(self.state_transition_probs[state][state_2])==list:
                            s+=self.state_transition_probs[state][state_2][age]
                        else:
                            s+=self.state_transition_probs[state][state_2]
                    self.state_transition_probs[state][state].append(1-s)
                if self.state_transition_probs[state][state] == [self.state_transition_probs[state][state][0]]*self.number_of_stratas:
                    self.state_transition_probs[state][state] = self.state_transition_probs[state][state][0]
        
        self.states_fix_period_time=states_fix_period_time
        if states_fix_period_time==None:
            self.states_fix_period_time= {}
            for state in states:
                self.states_fix_period_time[state] = 0
        
        self.quarantine_length = quarantine_length
        self.model = model
        self.spreading_function = spreading_function
        self.vaccination_method = vaccination_method
        self.vaccination_limit = vaccination_limit
        self.vaccination_start_time = vaccination_start_time
        self.vaccination_start_d_p = vaccination_start_d_p
        self.vaccination_efficient = vaccination_efficient
        self.vaccination_efficiency = vaccination_efficiency
        self.edge_size_limit = edge_size_limit
        self.is_graph_model = False
        self.exclude_died = exclude_died
        self.node_history_on = node_history_on
        self.testing_prob = testing_prob
        self.testing_capacity = testing_capacity
        self.school_closure=school_closure
        self.state_transition_validation_mtx = {}
        for state1 in states:
            self.state_transition_validation_mtx[state1]={}
            if state1 in self.state_transition_probs.keys():
                for state2 in states:
                    if state2 in self.state_transition_probs[state1].keys():
                        if type(self.state_transition_probs[state1][state2])==float or type(self.state_transition_probs[state1][state2])==int:
                            if self.state_transition_probs[state1][state2] > 0 :
                                self.state_transition_validation_mtx[state1][state2] = True
                        elif self.state_transition_probs[state1][state2] > [0]*len(self.state_transition_probs[state1][state2]) :
                            self.state_transition_validation_mtx[state1][state2] = True
                        else:
                            self.state_transition_validation_mtx[state1][state2] = False
                    else:
                        self.state_transition_validation_mtx[state1][state2] = False
            else:
                for state2 in states:
                    self.state_transition_validation_mtx[state1][state2] = False
        self.infectious_change_mtx=infectious_change_mtx
        if infectious_change_mtx==None:
            self.infectious_change_mtx={}
            for state1 in states:
                self.infectious_change_mtx[state1]={}
                for state2 in states:
                    if state1 == 'e':
                        if state2 == 'a' or state2 == 'i':
                            self.infectious_change_mtx[state1][state2] = 1
                        else:
                            self.infectious_change_mtx[state1][state2] = 0
                    elif state1 == 'a' or state1 =='i':
                        if state2 == 'r' or state2 =='d':
                            self.infectious_change_mtx[state1][state2] = -1
                        else:
                            self.infectious_change_mtx[state1][state2] = 0
                    else:
                        self.infectious_change_mtx[state1][state2] = 0
        self.states = states
        if self.spreading_function==None:
            self.spreading_function=spreading_function_lin
        if nodelist != None:
            self.sum_of_infectious = 0
            self.hypergraph = Hypergraph()
            for node in nodelist:
                self.add_node(node)
        
        
    
    def nodelist(self):
        '''
        Return with the nodelist of the hypergraph.
        '''
        return self.hypergraph.nodes
    
    def edgelist(self):
        '''
        Return with the edgelist of the hypergraph.
        '''
        return self.hypergraph.edges

    def get_nodes(self):
        '''
        Return with the actual nodelist of the hypergraph.
        '''
        nodes_ = []
        for node in self.nodelist():
            nodes_.append(node.get_node())
        return nodes_
    
    def get_edges(self):
        '''
        Return with the actual edgelist of the hypergraph.
        '''
        edges_ = []
        for edge in self.edgelist():
            edges_.append(edge.get_edge())
        return edges_

    def get_node_state(self, node):
        '''
        Parameters:
        node: Node object,
                a node of the hypergraph
              
        Returns:
        string
        viral state of the node
        '''
        return node.attr_list['state']

    def get_edge_type(self, edge):
        '''
        Parameters:
        edge: Hyperedge object,
                a hyperedge of the hypergraph
        
        Returns:
        string
        type of the hyperedge
        '''
        return edge.attr_list['type']
    
    def get_sum_of_infectious(self):
        '''
        Returns:
        sum of infected individuals in a population
        '''
        return self.sum_of_infectious

    def get_infection_rate(self):
        return self.sum_of_infectious/len(self.nodelist())
    
    def get_state_sum(self, _state):
        _sum = 0
        for node in self.nodelist():
            if node.attr_list['state'] == _state:
                _sum += 1
        return _sum
    
    def get_state_node_ids(self, _state):
        list_ = []
        for node in self.nodelist():
            if node.attr_list['state'] == _state:
                list_.append(node.id)
        
    def get_state_rate(self, _state):
        return self.get_state_sum(_state)/len(self.nodelist())
    
    def get_edge_infection_rate(self, edge):
        size = edge.size()
        if size == 0:
            return 0 
        else:
            return edge.attr_list['number of infectious'] / size 
        
    
    def get_neighbors(self, node, _type = None):
        
        """
        Get a set of neighboring nodes.
        Parameter
        ----------
        node_id: int, 
        The node's id to find the neighbors of.
        
        _type: string, optional, 
        The type of the neighbors corresponding to the hyperedge 
        type which connects the nodes.
        Returns
        -------
        neighbors : set
        A set of all nodes neighboring the queries node.
        """
        neighbor_nodes = set([])
        if _type == None:
            for edge in node.edgelist:
                neighbor_nodes.update(edge.nodelist)
        else:
            for edge in node.edgelist:
                if edge.attr_list['type'] == _type:
                    neighbor_nodes.update(edge.nodelist)
        if len(neighbor_nodes) > 0:    
            neighbor_nodes.remove(node)
        return neighbor_nodes   
    
    def degree(self, node):
        return node.degree()

    def add_node(self, _node):
        node = self.hypergraph.add_node(_node, True)
        if self.is_active(node):
            self.sum_of_infectious += 1
        #node.attr_list['weight']=self.state_transition_probs['id'][node.attr_list['age']]*(1/max(self.state_transition_probs['id']))
        #node.attr_list['weight']=self.state_transition_probs['i']['d'][node.attr_list['age']]*(1/max(self.state_transition_probs['i']['d']))
        for state in self.states:
            node.attr_list[state + ' time'] = 0
        
    def add_edge(self, _edge):
        if len(_edge[0])==0:
            pass
        else:
            s, hyperedge = self.hypergraph.add_edge(_edge, True, True)
            hyperedge.attr_list['number of infectious'] = s
            hyperedge.attr_list['weight'] = hyperedge.attr_list['spreading rate']/len(hyperedge.nodelist)

    def add_weights_to_the_nodes(self,weights=None, weight_function=None):
        if weights==None and weight_function != None:
            for node in self.nodelist():
                node.attr_list['weight'] = weight_function(node)
        elif len(weights)==len(self.nodelist):
            i=0
            for node in self.nodelist():
                node.attr_list['weight']=weights[i]
                i+=1
        elif len(weights)==self.number_of_stratas:
            for node in self.nodelist():
                node.attr_list['weight'] = weights[node.attr_list['age']]


    def expand_edge(self, edge, node):
        self.hypergraph.expand_edge(edge, node)
        if self.is_active(node):
            self.increase_number_of_infectious(edge, 1)
    
    def set_spreading_rate(self,rate,_type):
        for edge in self.edgelist():
            if edge.attr_list['type']==_type:
                edge.attr_list['spreading rate']=rate

    def set_appearance_prob(self,rate,_type):
        for edge in self.edgelist():
            if edge.attr_list['type']==_type:
                edge.attr_list['appearance probability']=rate
                
    def del_node(self,node):
        '''
        Delete a node from the hypergraph.
        Parameter:
        -----
        node: Node object 
        The node which have to be deleted
        '''
        self.hypergaph.del_node(node)
    
    def del_edge(self,edge):
        '''
        Delete a hyperedge from the hypergraph.
        Parameter:
        -----
        edge: Hyperedge object
        The hyperedge which have to be deleted
        '''
        self.hypergraph.del_edge(edge)
    
    def random_regrouping(self,number_of_edges):
        i=0
        while i < number_of_edges:
            edge=rnd.choice(self.hypergraph.edges)
            size=len(edge.nodelist)
            attr_list=edge.attr_list.copy()
            self.del_edge(edge)
            new_nodelist=np.random.choice(self.nodelist(),size = size, replace=False)
            self.add_edge([new_nodelist, attr_list])
            i+=1
        
    
    def generate_nodelists(self, stack, hyperedge_data):
        """
        Generate random family hyperedges from hyperedge distribution.
        parameter:
        -------
        _list: dict
        dictionary of the nodes
        
        """
        list_of_edges = []
        if hyperedge_data['distribution']=='uniform':
            for i in range(int(len(self.nodelist())/hyperedge_data['size'])):
                if (i+1)*hyperedge_data['size'] > len(self.nodelist()):
                    list_of_edges.append(self.nodelist()[(i*hyperedge_data['size']):len(self.nodelist())])
                else:
                    list_of_edges.append(self.nodelist()[(i*hyperedge_data['size']):(((i+1)*hyperedge_data['size']))])
            
        if hyperedge_data['distribution']=='poisson':
            mu = hyperedge_data['size mean']
            
            while stack != []:
                size = poisson.rvs(mu)
                if size>0:
                    family = []
                    if len(stack) < size:
                        family = list(stack)
                        for i in family:
                            stack.remove(i)
                    else:
                        i=0
                        while i < size:
                            member = rnd.choice(stack)
                            if member not in family:
                                stack.remove(member)
                                family.append(member)
                                i+=1
                    list_of_edges.append(family)
                
        elif hyperedge_data['distribution']=='binom':
            p = 1 - (hyperedge_data['size variance'] / hyperedge_data['size mean'])
            n = hyperedge_data['size mean'] / p
            
            while stack != []:
                size = np.random.binomial(n, p, 1)[0]
                if size>0:
                    family = []
                    if len(stack) < size:
                        family = list(stack)
                        for i in family:
                            stack.remove(i)
                    else:
                        for i in range(size):
                            member = rnd.choice(stack)
                            stack.remove(member)
                            family.append(member)
                    list_of_edges.append(family)
        return list_of_edges
    
    def generate_schools(self, hyperedge_data):
        '''Generates schools and classes inseide them.'''
        
        self.sort_by_age()
        students=[]
        i=0
        node=self.nodelist()[0]
        while node.attr_list['age']==0:
            students.append(node)
            i+=1
            node=self.nodelist()[i]
        schools = self.generate_nodelists(students, hyperedge_data[list(hyperedge_data.keys())[0]])
        rnd.shuffle(self.hypergraph.nodes[i:len(self.nodelist())])
        for school in schools:
            classrooms = self.generate_nodelists(students, hyperedge_data[list(hyperedge_data.keys())[1]])
            teachers = []
            for j in range(hyperedge_data['teachers per school']):
                if node.attr_list['age'] != 0:
                    teachers.append(node)
                    i+=1
                    node=self.nodelist()[i]
            classrooms.append(teachers)
            self.add_hyperedges_from_nodelists(classrooms,hyperedge_data[list(hyperedge_data.keys())[1]] ,list(hyperedge_data.keys())[1])
        self.add_hyperedges_from_nodelists(schools,hyperedge_data[list(hyperedge_data.keys())[0]] ,list(hyperedge_data.keys())[0])
        stack = []
        for node in list(self.nodelist()[i:len(self.nodelist())]):
            stack.append(node)
        list_of_nodelists = self.generate_nodelists(stack ,hyperedge_data[list(hyperedge_data.keys())[2]])
        self.add_hyperedges_from_nodelists(list_of_nodelists,hyperedge_data[list(hyperedge_data.keys())[2]] ,list(hyperedge_data.keys())[2])
    
    def generate_real_world_hyperedges(self, hyperedge_data, _type):
        pass
        '''
        Generate nodelists of hyperedges with the HyperPA method which provides small-world property to the hypergraph.
        
        parameter:
        -----
        prob: float between 0 and 1
        '''

        '''        
        
        self.add_hyperedge_from_nodelist(list(self.nodes.keys())[0:3], _type)
        i = 3
        for node_i in list(self.nodes.keys())[3:]:
            node = rnd.choice(list(self.nodes.keys())[0:i])
            if rnd.random() < _hyperedge_type['prob']:
                self.add_hyperedge_from_nodelist([node, node_i], _type)
            else:
                neighbors = list(self.get_neighbors(node, _type))
                #nodelists = self.generate_nodelists(neighbors, _hyperedge_type)
                #for nodelist in nodelists:
                neighbors.append(node_i)
                self.add_hyperedge_from_nodelist(neighbors, _type)
                
        sp = 1 - (hyperedge_data['size variance'] / hyperedge_data['size mean'])
        sn = hyperedge_data['size mean'] / p
        np = 1 - (hyperedge_data['new edge number variance'] / hyperedge_data['new edge number mean'])
        nn = hyperedge_data['new edge number mean'] / p
        for i in range(sn/2):
            nodelist = [list(self.nodelist())[2*i],[list(self.nodelist())[2*i+1]]]
            self.add_hyperedge_from_nodelist(nodelist,_type)
        
        for node in self.nodes.values():
        '''
    
    def preferential_choice(self, k, i, j,m,t):
        prob_int = [0]
        prob = 0
        for node in list(self.nodelist())[0:t-1]:
            p = self.degree(node)/(k*m*(t-k-1)+i*k+k)
            prob = prob + p
            prob_int.append(prob)
        r = rnd.random()
        index = bisect.bisect(prob_int, r) - 1
        return self.nodelist()[index]
                    
    
    def preferential_attachment_model(self, hyperedge_data, _type):
        k = hyperedge_data['edge size']
        m = hyperedge_data['number of new edges']
        self.add_hyperedge_from_nodelist(self.nodelist()[0:k],hyperedge_data, _type)
        t = k+1
        for node in self.nodelist()[k:]:            
            for i in range(m):
                new_hyperedge = [node]
                for j in range(k-1):    
                    node_ = self.preferential_choice(k,i,j,m,t)
                    while node_ in new_hyperedge:
                        node_ = self.preferential_choice(k,i,j,m,t)
                    new_hyperedge.append(node_)
                self.add_hyperedge_from_nodelist(new_hyperedge, hyperedge_data ,_type)
            t+=1
    
    def preferential_attachment_model_2(self, hyperedge_data, _type, model='BA'):
        k=0
        while k < 2:
            k = self.sample(hyperedge_data)
        m = hyperedge_data['number of new edges']
        self.add_hyperedge_from_nodelist(self.nodelist()[0:k],hyperedge_data, _type)
        if model=='BB':
            degree_vec=np.ones(k)
            i=0
            for node in self.nodelist()[:k]:
                degree_vec[i] = degree_vec[i]*node.attr_list['fitness']
                i += 1
        else:
            degree_vec = np.ones(k)
        node_indexes = np.arange(k,dtype=int)
        prob_vec = degree_vec/sum(degree_vec)
        t = k
        for node in self.nodelist()[k:]:
            for i in range(m):
                size=0
                while size<1:
                    size = self.sample(hyperedge_data)
                if t < size:
                    size=t
                new_hyperedge = [node]
                new_nodes_inds = np.random.choice(node_indexes,size= size-1, replace=False, p= prob_vec)
                for index in  new_nodes_inds:
                    selected_node = self.hypergraph.nodes[index]
                    if model=='BB':
                        degree_vec[index]+=selected_node.attr_list['fitness']
                    else:
                        degree_vec[index]+=1
                    new_hyperedge.append(selected_node)
                self.add_hyperedge_from_nodelist(new_hyperedge, hyperedge_data ,_type)
                prob_vec = degree_vec/sum(degree_vec)
            node_indexes=np.append(node_indexes,t)
            if model=='BB':
                degree_vec=np.append(degree_vec,m*node.attr_list['fitness'])
            else:
                degree_vec=np.append(degree_vec,m)
            prob_vec = degree_vec/sum(degree_vec)
            t += 1
    
    def simple_preferential_attachment_model(self, hyperedge_data, _type):
        k=0
        while k < 2:
            k = self.sample(hyperedge_data)
        m = hyperedge_data['number of new edges']
        self.add_hyperedge_from_nodelist(self.nodelist()[0:k],hyperedge_data, _type)
        self.degree
        t = k+1
        for node in self.nodelist()[k:]:
            new_hyperedges = self.hyperedges_from_preferential_choice(node,m,t,hyperedge_data)
            self.add_hyperedges_from_nodelists(new_hyperedges,hyperedge_data ,_type)
            t+=1
        
    def hyperedges_from_preferential_choice(self,_node,m,t, hyperedge_data):
        prob_int = [0]
        prob = 0
        if 'fitness' in _node.attr_list.keys():
            s= np.matmul([node.attr_list['fitness'] for node in self.nodelist()],[self.degree(node) for node in self.nodelist()])
        else:
            s = sum([self.degree(node) for node in self.nodelist()])
        for node in list(self.nodelist())[0:t-1]:
            if 'fitness' in _node.attr_list.keys():
                p = self.degree(node)*node.attr_list['fitness']/s
            else:
                p = self.degree(node)/s
            prob = prob + p
            prob_int.append(prob)
        prob_int=np.array(prob_int)*(1/prob_int[-1])
        new_hyperedges = []
        for i in range(m):
            new_hyperedge = [_node]
            k=0
            while k < 2:
                k = self.sample(hyperedge_data)
            for i in range(k-1):
                r = rnd.random()
                index = bisect.bisect(prob_int, r) - 1
                new_node_to_append=self.hypergraph.nodes[index]
                if new_node_to_append not in new_hyperedge:
                    new_hyperedge.append(new_node_to_append)
            new_hyperedges.append(new_hyperedge)
        return new_hyperedges
    
    def bianconi_barabasi_model(self, hyperedge_data, _type, node_fitness=None):
        k=0
        while k < 2:
            k = self.sample(hyperedge_data)
        m = hyperedge_data['number of new edges']
        self.add_hyperedge_from_nodelist(self.nodelist()[0:k],hyperedge_data, _type)
        t = k+1
        if node_fitness != None:
            i=0
            for node in self.nodelist():
                node.attr_list['fitness']=node_fitness[i]
                i+=1
            
        for node in self.nodelist()[k:]:
            new_hyperedges = self.hyperedges_from_preferential_choice(node,m,t,hyperedge_data)
            self.add_hyperedges_from_nodelists(new_hyperedges,hyperedge_data ,_type)
            t+=1
            
            
    def add_hyperedge_from_nodelist(self, nodelist, hyperedge_data, _type):
        _attr_list = {'type' : _type, 'number of infectious' : 0, 'spreading rate' : hyperedge_data['spreading rate']}
        if _type=='e' and self.model=='realistic':
            _attr_list['appearance probability'] = hyperedge_data['appearance probability']
        self.add_edge([nodelist, _attr_list])         
    
    def add_hyperedges_from_nodelists(self, _list_of_nodelists, hyperedge_data , _type):
        for nodelist in _list_of_nodelists:
            self.add_hyperedge_from_nodelist(nodelist, hyperedge_data, _type)
    
    def generate_hyperedges(self, _hyperedge_type, _type):
        stack = []
        self.sort_random()
        for node in list(self.nodelist()):
            stack.append(node)
        list_of_nodelists = self.generate_nodelists(stack ,_hyperedge_type)
        self.add_hyperedges_from_nodelists(list_of_nodelists,_hyperedge_type ,_type)
        
    def generate_Erdos_Renyi_hg(self, _hyperedge_type, _type):
        _nodelists = []
        i = 0
        while i < _hyperedge_type['number of edges']:
            _nodelist=[]
            j = 0
            while j < _hyperedge_type['size']:
                _node=rnd.choice(self.nodelist())
                if _node not in _nodelist:
                    _nodelist.append(_node)
                    j += 1
            if _nodelist not in _nodelists:
                _nodelists.append(_nodelist)
                i += 1
        self.add_hyperedges_from_nodelists(_nodelists, _hyperedge_type, _type)

    def compartmental_model(self, hyperedge_type, _type):
        stubs=[]
        node_ind = 0
        for node_degree in hyperedge_type['degree distribution']:
            for i in range(node_degree):
                stubs.append(self.hypergraph.nodes[node_ind])
            node_ind += 1
        rnd.shuffle(stubs)
        list_of_nodelists = self.generate_nodelists(stubs, hyperedge_type)
        self.add_hyperedges_from_nodelists(list_of_nodelists, hyperedge_type,_type)
    
    def sample(self, distribution_data):
        '''
        Returns one sample element from the distribution defined in the distribution_data.

        The distribution_data can be given as binomial, poisson, uniform, 
        or a finite distribution vector.
        '''
        if distribution_data['distribution']=='binom':
            p = 1 - (distribution_data['size variance'] / distribution_data['size mean'])
            n = distribution_data['size mean'] / p
            return np.random.binomial(n, p, 1)[0]
        
        elif distribution_data['distribution']=='poisson':
            mu = distribution_data['size mean']
            return poisson.rvs(mu)

        elif distribution_data['distribution']=='gamma':
            a = distribution_data['a']
            scale = distribution_data['scale']
            loc = distribution_data['loc']
            return int(gamma.rvs(a,scale=scale,loc=loc))
        
        elif distribution_data['distribution']=='pareto':
            b = distribution_data['b']
            scale = distribution_data['scale']
            loc = distribution_data['loc']
            return int(pareto.rvs(b,scale=scale,loc=loc))

        elif distribution_data['distribution']=='uniform':
            return distribution_data['size']

        else:
            r = rnd.random()
            return bisect.bisect(distribution_data['distribution'],r)-1


    def node_weight_attachment_model_connected(self, hyperedge_type, _type):
        self.sort_random()
        for node in self.nodelist():
            new_hyperedge = [node]
            size = self.sample(hyperedge_type)
            i = 0
            while i < size:
                node_ind = self.sample(hyperedge_type['node weight'])
                node_=self.nodelist()[node_ind]
                if node_ not in new_hyperedge:
                    new_hyperedge.append(node_)
                    i += 1
            self.add_hyperedge_from_nodelist(new_hyperedge, hyperedge_type ,_type)
    
    def node_weight_attachment_model(self, hyperedge_type, _type):
        self.sort_random()
        for j in range(hyperedge_type['number of edges']):
            new_hyperedge = []
            size = self.sample(hyperedge_type)
            i = 0
            while i < size:
                node_ind = self.sample(hyperedge_type['node weight'])
                node_=self.nodelist()[node_ind]
                if node_ not in new_hyperedge:
                    new_hyperedge.append(node_)
                    i += 1
            self.add_hyperedge_from_nodelist(new_hyperedge, hyperedge_type ,_type)
                

        
    def get_exposed(self, node):
        if node.attr_list['state'] is not 's':
            print('The person is not susceptible to the virus.')
        else:
            node.attr_list['state'] = 'e'
    
    def get_asymptotic(self, node):
        if node.attr_list['state'] is not 'e':
            print('The person is not exposed to the virus.')
        else:
            node.attr_list['state'] = 'a'            
    
    def get_infected(self, node):
        if node.attr_list['state'] is not 'e':
            print('The person is not exposed to the virus.')
        else:
            node.attr_list['state'] = 'i'
    
    def recover(self, node):
        if node.attr_list['state'] == 'i' or node.attr_list['state'] == 'a' or node.attr_list['vaccination period'] != None:
            node.attr_list['state'] = 'r'
        else:
            print('The person is not infected by the virus.')
    
    def die(self, node):
        if node.attr_list['state'] != 'i':
            print('The person is not heavily infected by the virus.')
        else:
            node.attr_list['state'] = 'd'
            if self.exclude_died:
                self.exclude_from_all(node)
    
    
    def transition_to_state(self, node, new_state):
        '''
        Parameters:
            node: Node object
            new_state: state of the new state of the node
        '''
        validate = self.state_transition_validation_mtx[node.attr_list['state']][new_state]
        if validate:
            change = self.infectious_change_mtx[node.attr_list['state']][new_state]
            if change != 0:
                self.change_infected_numbers(node, change)
            node.attr_list['state'] = new_state
            
        else:
            raise NameError('It is not valid state transition: '+ node.attr_list['state']+' to '+ new_state )
            
    def test(self, node):
        if self.is_active(node) and node.attr_list['is tested']==0:
            node.attr_list['is tested'] = 1
            return 1
        return 0
            
    def increase_quarantine_time(self, node):
        if node.attr_list['quarantine time'] == 0:
            node.attr_list['fresh quarantine'] = True
        node.attr_list['quarantine time'] += 1
    
    def fresh_quarantine_over(self, node):
        node.attr_list['fresh quarantine'] = False
    
    def is_active(self, node):
        return node.attr_list['state'] == 'i' or node.attr_list['state'] == 'a'
    
    def increase_state_time(self, node, state):
        if (state +' time') not in node.attr_list.keys():
            node.attr_list[state +' time'] = 0
        else:
            node.attr_list[state +' time'] += 1
        
    
    def get_type(self, edge):
        return edge.attr_list['type']
    
    def increase_number_of_infectious(self, edge, additional_infections):
        edge.attr_list['number of infectious'] = edge.attr_list['number of infectious'] + additional_infections
        
    def spreading_function_lin(self, size, number_of_infectious):
      return number_of_infectious/size

    def spread_through_edge(self,edge):
        """
        The virus infects one person through a hyperedge with 
        probabilty = spreading rate of the edge * number of active infectious persons in the hyperedge / size of the hyperedge.
        The person will be exposed if the virus infects him or her.
        Parameter:
        -----
        edge: Hyperedge object, the hyperedge
        """
        infections = 0
        if  edge.size() < 2:
            pass
        elif edge.attr_list['type'] != 'f' and edge.size() > self.edge_size_limit:
            pass
        elif self.school_closure and (edge.attr_list['type']=='s' or edge.attr_list['type']=='c'):
            pass
        else:
            inf_prob = edge.attr_list['spreading rate'] * self.spreading_function(edge.size(),edge.attr_list['number of infectious'])
            for node in edge.nodelist:
                pi=1
                if 'immunation rate' in node.attr_list.keys():
                    pi=node.attr_list['immunation rate']
                if rnd.random() < inf_prob*pi and self.get_node_state(node) == 's':
                    self.get_exposed(node)
                    infections += 1
        return infections
        
    def change_infected_numbers(self, node, n):
        if node.attr_list['quarantine time'] > 0:
            for edge in node.edgelist:
                if edge.attr_list['type'] == 'f':
                    self.increase_number_of_infectious(edge, n)
        else:
            for edge in node.edgelist:
                self.increase_number_of_infectious(edge, n)
        self.sum_of_infectious += n
    
    def state_transition(self,node):
        """
        The person change viral state with an exact probability depending on its state 
        at the begining of the timestep.
        
        The self.state_transition dict must have the following structure:
        
        - keys: state1 names
        - values: dictionaries in the following form:   -keys: state2 names 
                                                        -values: state1 -> state2 transition probability
        
        Parameter:
        ----
        node: Person object
        """
        if self.model=='basic' or self.model=='realistic':
            state = self.get_node_state(node)
            if self.number_of_stratas > 1:
                age = node.attr_list['age']
            if state=='d' or state=='r':
                return 0
            elif node.attr_list[state+' time'] < self.states_fix_period_time[state]:
                self.increase_state_time(node, state)
            else:
                transition_probs = []
                for state_2 in self.state_transition_probs[state].keys():
                        if type(self.state_transition_probs[state][state_2])==list:
                            transition_probs.append(self.state_transition_probs[state][state_2][age])
                        else:
                            transition_probs.append(self.state_transition_probs[state][state_2])
                next_state = rnd.choices(list(self.state_transition_probs[state].keys()),weights=transition_probs)[0]
                if next_state==state:
                    self.increase_state_time(node, state)
                else:
                    if state == 'h':
                        self.place_back_to_all(node)
                    self.transition_to_state(node,next_state)
                    if next_state == 'h':
                        self.exclude_from_all(node)
                        if node.attr_list['is tested']==0:
                            node.attr_list['is tested'] = 1
                            return 1
            return 0
                
            
        if self.model=='retro':
            if self.get_node_state(node) == 'e':
                r = rnd.random()
                if type(self.state_transition_probs['ei'])==list:
                    if r < self.state_transition_probs['ei'][node.attr_list['age']]:
                        self.get_infected(node)
                        self.change_infected_numbers(node, 1)
                    elif r < (self.state_transition_probs['ei'][node.attr_list['age']] + self.state_transition_probs['ea'][node.attr_list['age']]):
                        self.get_asymptotic(node)
                        self.change_infected_numbers(node, 1)
                else:
                    if r < self.state_transition_probs['ei']:
                        self.get_infected(node)
                        self.change_infected_numbers(node, 1)
                    elif r < (self.state_transition_probs['ei'] + self.state_transition_probs['ea']):
                        self.get_asymptotic(node)
                        self.change_infected_numbers(node, 1)
            elif self.get_node_state(node) == 'i':
                if rnd.random() < self.state_transition_probs['ird']:
                    if rnd.random() < self.state_transition_probs['id'][node.attr_list['age']]:
                        self.die(node)
                    else:
                        self.recover(node)
                    self.change_infected_numbers(node, -1)
            elif self.get_node_state(node) == 'a':
                if rnd.random() < self.state_transition_probs['ar']:
                    self.recover(node)
                    self.change_infected_numbers(node, -1)
    
    def exclude_from_work(self, node):
        for edge in node.edgelist:
            if edge.attr_list['type'] != 'f':
                if node not in edge.nodelist:
                    print(node.attr_list['id'])
                edge.nodelist.remove(node)
                if self.is_active(node):
                    self.increase_number_of_infectious(edge,-1)
    
    def place_back_to_all(self, node):
        for edge in node.edgelist:
            edge.nodelist.append(node)
            if self.is_active(node):
                self.increase_number_of_infectious(edge,1)
                    
    def exclude_from_all(self, node):
        for edge in node.edgelist:
            if node in edge.nodelist:
                edge.nodelist.remove(node)
                if self.is_active(node):
                    self.increase_number_of_infectious(edge,-1)
    
    def quarantine(self, node):
        if node.attr_list['quarantine time'] == 0 and self.is_active(node):
            family = [node]
            family.extend(self.get_neighbors(node, 'f'))
            for member in family:
                self.increase_quarantine_time(member)
                self.exclude_from_work(member)
                
    def quarantine_checking(self, node):
        if node.attr_list['quarantine time'] == self.quarantine_length:
            node.attr_list['quarantine time'] = 0
            for edge in node.edgelist:
                if edge.attr_list['type'] != 'f':
                    edge.nodelist.append(node)
                    if self.is_active(node):
                        self.increase_number_of_infectious(edge, 1)
        elif node.attr_list['quarantine time'] > 0:
            self.increase_quarantine_time(node)
        
        else:
             pass   
    
    def vaccinate_node(self, node):
        node.attr_list['vaccination period'] = 0

    def sort_by_age(self):
        hash_map = {}
        nodes_=[]
        for node in self.nodelist():
            if node.attr_list['age'] not in hash_map.keys():
                hash_map[node.attr_list['age']] = [node]
            else:
                hash_map[node.attr_list['age']].append(node)
        for key in sorted(hash_map.keys()):
            nodes_ = nodes_ + hash_map[key]
        self.hypergraph.nodes = nodes_

    def sort_random(self):
        rnd.shuffle(self.hypergraph.nodes)

    def bubble_sort_by_degree(self): 
        n = len(self.nodelist()) 
        for i in range(n): 
            s = 0
            for j in range(0, n-i-1): 
                if self.nodelist()[j].degree() < self.nodelist()[j+1].degree(): 
                    self.hypergraph.nodes[j], self.hypergraph.nodes[j+1] = self.hypergraph.nodes[j+1], self.hypergraph.nodes[j]
                    s += 1
            if s ==0:
                return
    
    def sort_by_degree(self):
        self.hypergraph.nodes.sort(key=lambda x: x.degree())
    
    def sort_by_weighted_degree(self):
        self.hypergraph.nodes.sort(key=lambda x: x.weighted_degree())
    
    def sort_by_spectral_centrality(self, mode):
        importanc_vec,edge_importance_vec = self.hypergraph.spectral_centrality(mode, graph_exp=True)
        i=0
        for node in self.hypergraph.nodes:
            node.attr_list['centrality']=importanc_vec[i]
            i+=1
        self.hypergraph.nodes.sort(key=lambda x: x.attr_list.get('centrality'))
    
    def sort_by_pagerank(self):
        importanc_vec,edge_importance_vec = self.hypergraph.PageRank()
        i=0
        for node in self.hypergraph.nodes:
            node.attr_list['centrality']=importanc_vec[i]
            i+=1
        self.hypergraph.nodes.sort(key=lambda x: x.attr_list.get('centrality'))
    
    def change_vaccination_time(self,node):
        if node.attr_list['vaccination period'] != None:
            node.attr_list['vaccination period'] += 1
        if node.attr_list['vaccination period'] == 14:
            if self.vaccination_efficient=='total efficient':
                if self.is_active(node):
                    self.change_infected_numbers(node, -1)
                self.recover(node)
            elif self.vaccination_efficient=='with immunizing rate':
                if not self.is_active(node):
                    if rnd.random()<self.vaccination_efficiency:
                        self.recover(node)
            elif self.vaccination_efficient=='with lower spreading rate':
                node.attr_list['immunation rate'] = 1-self.vaccination_efficiency
    
  
    def sort_for_vaccination(self):
        if self.vaccination_method == 'random':
            self.sort_random()
        if self.vaccination_method == 'age':
            self.sort_by_age()
        if self.vaccination_method == 'degree':
            self.sort_by_degree()
        if self.vaccination_method == 'weighted degree':
            self.sort_by_weighted_degree()
        if self.vaccination_method == 'spectral centrality bipartite':
            self.sort_by_spectral_centrality('bipartite')
        if self.vaccination_method == 'spectral centrality tudisco':
            self.sort_by_spectral_centrality('tudisco')
        if self.vaccination_method == 'PageRank':
            self.sort_by_pagerank()

    def timestep_end(self):
        for node in self.nodelist():
            self.fresh_quarantine_over(node)
  
    def timestep(self, step_index,vacc_start=False):
        '''
        Make one timestep in the simulation.
        '''
        vacc_ind = 0
        daily_infections = 0
        daily_first_positive_tests = 0
        if self.testing_prob != 0:
            test_ind = 0
        for edge in self.edgelist():
            infections = 0
            if edge.attr_list['type']=='e' and self.model=='realistic':
                if rnd.random() < edge.attr_list['appearance probability']:
                    infections = self.spread_through_edge(edge)
            else:
                infections = self.spread_through_edge(edge)
            daily_infections += infections
        for node in self.nodelist():
            if node.attr_list['state'] != 'd':
                if node.attr_list['state']!= 's':
                    h = self.state_transition(node)
                    daily_first_positive_tests += h
                if self.vaccination_method != None and vacc_start: 
                    self.change_vaccination_time(node)
                    if vacc_ind < self.vaccination_limit +1 :
                        if not node.attr_list['is tested'] and node.attr_list['vaccination period']==None:
                            self.vaccinate_node(node)
                            vacc_ind += 1
                if self.testing_prob != 0:
                    result = 0
                    if type(self.testing_prob) == int or type(self.testing_prob) == float:
                        if rnd.random() < self.testing_prob:
                            result = self.test(node)
                            self.quarantine(node)
                            test_ind += 1
                    elif  rnd.random() < self.testing_prob[self.get_node_state(node)] and test_ind < self.testing_capacity:
                        result = self.test(node)
                        self.quarantine(node)
                        test_ind += 1
                    daily_first_positive_tests += result
            if not node.attr_list['fresh quarantine']:
                self.quarantine_checking(node)  
        self.timestep_end()
        return daily_infections, daily_first_positive_tests
    
    
    def run(self, steps=None):
        '''
        Run the simulation of the Epidemic spreading for a given number of timesteps.
        Parameter: 
        -----
        steps: int, number of timesteps to be done
        Output: 
        -----
        infected_rates: list, The rates of the infected people and the non infected people at each timesteps
                                 ordered in a tuple.
        '''
        
        rates = {}
        rates['infectious'] = [self.get_infection_rate()]
        for state in self.states:
            rates[state] = [self.get_state_rate(state)]
        self.node_history = [self.get_nodes()]
        daily_infections, daily_positive_tests = [0], [0]
        i = 0
        vacc_on=False
        if self.vaccination_method != None:
            self.sort_for_vaccination()
        if steps==None:
            while self.get_infection_rate()>0 or self.get_state_rate('e')>0:
                d_i,d_p=self.timestep(i, vacc_on)
                if d_p > self.vaccination_start_d_p and i > self.vaccination_start_time:
                    vacc_on=True
                daily_infections.append(d_i)
                daily_positive_tests.append(d_p)
                rates['infectious'].append(self.get_infection_rate())
                for state in self.states:
                    rates[state].append(self.get_state_rate(state))
                if self.node_history_on:
                    self.node_history.append(self.get_nodes())
                #self.check_infectious_number_validity()
                i += 1
        
        else:  
            for i in range(steps):
                d_i,d_p=self.timestep(i,vacc_on)
                if d_p > self.vaccination_start_d_p and i > self.vaccination_start_time:
                    vacc_on=True
                daily_infections.append(d_i)
                daily_positive_tests.append(d_p)
                rates['infectious'].append(self.get_infection_rate())
                for state in self.states:
                    rates[state].append(self.get_state_rate(state))
                if self.node_history_on:
                    self.node_history.append(self.get_nodes())
                #self.check_infectious_number_validity()
                i += 1
        return rates, daily_infections, daily_positive_tests, i
    
    def end_quarantines_vaccination_death(self):
        for node in self.nodelist():
            node.attr_list['vaccination period'] = None
            if node.attr_list['quarantine time'] > 0:
                node.attr_list['quarantine time'] = 0
                for edge in node.edgelist:
                    if edge.attr_list['type'] != 'f':
                        edge.nodelist.append(node)
                        if self.is_active(node):
                            self.increase_number_of_infectious(edge, 1)
            if self.exclude_died:
                if node.attr_list['state']== 'd':
                    for edge in node.edgelist:
                        edge.nodelist.append(node)
            if node.attr_list['state']== 'h':
                    for edge in node.edgelist:
                        edge.nodelist.append(node)

    def reset(self):
        self.end_quarantines_vaccination_death()
        self.sum_of_infectious = 0
        i = 0
        for node in self.nodelist():
            node.set_attr_list(self.node_history[0][i]['attr list'])
            if self.is_active(node):
                self.sum_of_infectious += 1
            i += 1
        for edge in self.edgelist():
            s = 0
            for node in edge.nodelist:
                if self.is_active(node):
                    s += 1
            edge.attr_list['number of infectious'] = s
        
    def hypergraph_to_graph(self):
        self.hypergraph.to_graph()
        self.is_graph_model = True
        self.spreading_function = spreading_function_lin
    
    def check_infectious_number_validity(self):
        for edge in self.edgelist():
            s=0
            for node in edge.nodelist:
                if self.is_active(node):
                    s += 1
            if edge.attr_list['number of infectious'] != s:
                print('ajaj')
                
    def degree_distrubution(self):
        return self.hypergraph.degree_distribution()

def probs_to_distribution_function(probs):
    s=0
    distr_array=[0]
    for p in probs:
        s+=p
        distr_array.append(s)
    if s!=1:
        raise NameError('probs was not a probabilty vector, i.e. sum(probs) not equal to 1')
    return distr_array
    
    
def spreading_function_lin(size, number_of_infectious):
    return number_of_infectious/(size-1)
    