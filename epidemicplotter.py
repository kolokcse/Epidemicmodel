import numpy as np
import random as rnd
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class Eplot(object):
    
    def __init__(self):
        pass
    
    def av_cut_10_percent(self,vec):
        l = len(vec)
        vec.sort()
        return sum(vec[int(l/10):int(l*9/10)])/len(vec[int(l/10):int(l*9/10)])
        
    def plotting_av_rates(self,run_hg, filename=None):
        '''
        Plots average infectious,s,r,d rates of many runs. Cuts 10% high and low.
        '''
        av_rates_hg = {}
        for state in run_hg[0].keys():
            av_hg = []
            for j in range(len(run_hg[0]['infectious'])):
                av = self.av_cut_10_percent([run_hg[k][state][j] for k in range(len(run_hg))])
                av_hg.append(av)
            av_rates_hg[state]=av_hg

        plt.figure(figsize=(10,7))
        plt.plot(av_rates_hg['infectious'], 'r')
        plt.plot(av_rates_hg['s'], 'b')
        plt.plot(av_rates_hg['r'], 'g')
        plt.plot(av_rates_hg['d'], 'k')
        plt.ylabel('rates', fontsize=25)
        plt.xlabel('timestep', fontsize=25)

        red_line = mlines.Line2D([], [], color='r', label='infectious')
        blue_line = mlines.Line2D([], [], color='b', label='susceptible')
        green_line = mlines.Line2D([], [], color='g', label='recovered')
        black_line = mlines.Line2D([], [], color='k', label='dead')  
        plt.legend(handles=[blue_line, red_line, green_line,black_line], fontsize='xx-large', loc='upper right')
        if filename != None:
            plt.savefig(filename, format='png')
        plt.show()
        
        
        
    def plotting_av_rates_multicompare(self,runs, modes, curve='i',runsg=None ,filename=None):
        '''Plots av rates of the curve state of the simulations in the runs.'''
        cols=['r','b','g','k','y']
        colsg=['r--','b--','g--','k--','y--']
        av_rates={}
        if runsg!= None:
            av_ratesg={}
        for x in runs.keys():
          av_rates[x]=[]
          if runsg!= None:
              av_ratesg[x]=[]
        for i in range(len(runs[modes[0]][0][curve])):
            for x in runs.keys():
                if runsg!= None:
                    avg = self.av_cut_10_percent([runsg[x][j][curve][i] for j in range(len(runsg[x]))])
                    av_ratesg[x].append(avg)
                av = self.av_cut_10_percent([runs[x][j][curve][i] for j in range(len(runs[x]))])
                av_rates[x].append(av)

        plt.figure(figsize=(10,7))
        for i in range(len(modes)):
            plt.plot(av_rates[modes[i]],cols[i],label=modes[i])
            if runsg!= None:
                plt.plot(av_ratesg[modes[i]],colsg[i])
        plt.ylabel('av. '+ curve +' rates', fontsize=25)
        plt.xlabel('timestep', fontsize=25)
        
        plt.legend(fontsize='x-large', loc='center right')
        plt.show()
        if filename != None:
            plt.savefig(filename, format='png')
        return [av_rates[modes[i]][-1] for i in range(len(modes))]
    
    def plot_to_compare_2(self,rates_hg,rates_g,label_1,label_2,state='infectious',filename=None):
        plt.figure(figsize=(10,7))
        max_rates_hg = []
        min_rates_hg = []
        av_rates_hg = []
        max_rates_g = []
        min_rates_g = []
        av_rates_g = []
        for i in range(len(rates_g[0][state])):
            max_hg = 0
            min_hg = 1
            max_g = 0
            min_g = 1
            shg = 0
            sg = 0
            for j in range(len(rates_g)):
                if rates_hg[j][state][i] > max_hg:
                    max_hg = rates_hg[j][state][i]
                if rates_hg[j][state][i] < min_hg:
                    min_hg = rates_hg[j][state][i]
                shg = shg + rates_hg[j][state][i]
                if rates_g[j][state][i] > max_g:
                    max_g = rates_g[j][state][i]
                if rates_g[j][state][i] < min_g:
                    min_g = rates_g[j][state][i]
                sg = sg + rates_g[j][state][i]
                max_rates_hg.append(max_hg)
                min_rates_hg.append(min_hg)
                av_rates_hg.append(shg/len(rates_g))
                max_rates_g.append(max_g)
                min_rates_g.append(min_g)
                av_rates_g.append(sg/len(rates_g))
  
        x = range(len(rates_g[0][state]))
        plt.fill_between(x,  max_rates_hg,min_rates_hg, color='C1', alpha=0.5)
        plt.fill_between(x,  max_rates_g,min_rates_g, color='C0', alpha=0.5)
        plt.plot(x,av_rates_hg,'r')
        plt.plot(x,av_rates_g,'b')
        hg_line = mlines.Line2D([], [], color='r', label=label_1)
        g_line = mlines.Line2D([], [], color='b', label=label_2) 
        plt.ylabel('rate of '+ state, fontsize=25)
        plt.xlabel('timestep', fontsize=25)
        plt.legend(handles=[hg_line, g_line], fontsize='xx-large', loc='upper right')
        if filename != None:
            plt.savefig(filename, format='png')
        plt.show()
        
    def plot_1_rates(self,rates, filename=None):  
        plt.figure(figsize=(10,7))
        plt.plot(rates['infectious'], 'r')
        plt.plot(rates['s'], 'b')
        plt.plot(rates['r'], 'g')
        plt.plot(rates['d'], 'k')
        if 'h' in rates.keys():
            plt.plot(rates['h'], 'm')
        plt.ylabel('rates', fontsize=25)
        plt.xlabel('timestep', fontsize=25)
        blue_line = mlines.Line2D([], [], color='b', label='susceptible')
        red_line = mlines.Line2D([], [], color='r', label='infectious')
        green_line = mlines.Line2D([], [], color='g', label='recovered')
        black_line = mlines.Line2D([], [], color='k', label='dead')  
        if 'h' in rates.keys():
            m_line = mlines.Line2D([], [], color='m', label='hospitalized')  
            plt.legend(handles=[blue_line, red_line, green_line, black_line, m_line], fontsize='xx-large', loc='center right')
        else:
            plt.legend(handles=[blue_line, red_line, green_line, black_line], fontsize='xx-large', loc='center right')
        if filename != None:
            plt.savefig(filename, format='png')
        plt.show()
        
    def final_size_distribution_compare(self,rates_1,rates_2,label_1, label_2, size, _scale='lin'):
        distr_1 = [0]
        for run in rates_1:
            final_size = int((run['r'][-1] + run['d'][-1]- run['infectious'][0])*size)
            if len(distr_1) < final_size + 1:
                for i in range(final_size + 1-len(distr_1)):
                    distr_1.append(0)
            distr_1[final_size] +=1
        distr_2 = [0]
        for run in rates_2:
            final_size = int((run['r'][-1] + run['d'][-1]- run['infectious'][0])*size)
            if len(distr_2) < final_size + 1:
                for i in range(final_size + 1-len(distr_2)):
                    distr_2.append(0)
            distr_2[final_size] +=1
        maxlen=max(len(distr_1),len(distr_2))
        distr_1 += [0] * (maxlen-len(distr_1))
        distr_2 += [0] * (maxlen-len(distr_2))
        x = np.arange(maxlen)
        width = 0.4
        plt.figure(figsize=(15,6))
        
        plt.legend()
        plt.tight_layout()
        if _scale=='log':
            plt.yscale('log')
            plt.xscale('log')
            plt.plot(x, distr_1, 'r*', label=label_1)
            plt.plot(x, distr_2, 'g^', label=label_2)
        else:
            plt.bar(x - width/2, distr_1, width, label=label_1)
            plt.bar(x + width/2, distr_2, width, label=label_2)
        plt.ylabel('count', fontsize=15)
        plt.xlabel('final size', fontsize=15)
        plt.show()
        
    def plot_distributions(self, distrs, labels, figsize=(10,6), scale='lin'):
        cols=['r*','bo','g^','kD','mv']
        plt.figure(figsize=figsize)
        if scale=='lin':
            plt.bar(range(len(distrs)),distrs)
        elif scale == 'log':
            i=0
            for distr in distrs:
                plt.plot(distr,cols[i], label=labels[i])
            plt.yscale('log')
            plt.xscale('log')
        plt.xlabel(label, fontsize=15)
        plt.show()
    
    def plotting_to_compare_rates(self, rates, labels, curve, filename=None):
        cols=['r-*','b-o','g-^','k-D','m-v']
        plt.figure(figsize=(10,7))
        for i in range(len(rates)):
            plt.plot(rates[i][curve], label = labels[i])
        plt.ylabel('rates', fontsize=25)
        plt.xlabel('timestep', fontsize=25)
        plt.legend(fontsize='xx-large', loc='center right')
        if filename != None:
            plt.savefig(filename, format='png')
        plt.show()
        
    def av_numbers_by_keyparameter(self, runs,labels, scat=False):
        cols=['r-*','b-o','g-^','k-D','m-v']
        modes=list(runs.keys())
        keys=list(runs[modes[0]].keys())
        plt.figure(figsize=(10,7))
        if scat:
            alpha=0.5
            x = [[int(j) for e in runs[modes[0]][j]] for j in keys]
            for i in modes:
                y= [runs[i][j] for j in keys]
                plt.scatter(x,y, alpha=alpha)
        for i in range(len(modes)):
            x = np.array(keys, dtype=float)
            y= [self.av_cut_10_percent(runs[modes[i]][j]) for j in keys]
            plt.plot(x,y,cols[i],label=modes[i])
        plt.legend(fontsize='xx-large')
        plt.ylabel(labels[0], fontsize=15)
        plt.xlabel(labels[1], fontsize=15)
        plt.show()