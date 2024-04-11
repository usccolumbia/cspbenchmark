import ipywidgets as widgets # New
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import os

from agox.helpers.plot_confinement import plot_atoms, plot_cell

FONTCOLOR = 'black'

def make_save_folder():
    today = date.today().strftime('%d_%m_%y')
    folder = os.path.expanduser('~')+'/figs/'+today + '/'

    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder

def savefig(path, fig):
    if not os.path.exists(path):
        fig.savefig(path, bbox_inches='tight')
    else:
        print('Warning: Figure with that name already exists')

####################################################################################################################
# Interactive: Functions for interactive Jupyter notebook plots
####################################################################################################################

class InteractiveStructureHistogram:

    def __init__(self, analysis, xlim=None, ylim=None, template_indices=None, histogram_kwargs=None):

        # Attached 'Analysis' object:
        self.analysis = analysis

        # Get all best structures and their energies:
        self.structures, self.energies = self.analysis.get_best_structures()

        self.first_call = True

        self.xlim = xlim
        self.ylim = ylim

        self.histogram_kwargs = histogram_kwargs
        self.template_indices = template_indices

    def update_plot(self, index):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        #self.analysis.plot_structure_nice(ax[0], self.structures[index])
        if self.template_indices is None:
            plot_atoms(self.structures[index], ax=ax[0])
        else:
            plot_atoms(self.structures[index][self.template_indices], ax=ax[0], alpha=0.5)
            structure_indices = [i for i in range(len(self.structures[index])) if i not in self.template_indices]        
            plot_atoms(self.structures[index][structure_indices], ax=ax[0])

        plot_cell(ax[0], self.structures[index].cell, np.array([0, 0, 0]), linestyle='--', color='black')
        if self.xlim is not None:
            ax[0].set_xlim(self.xlim)
        if self.ylim is not None:
            ax[0].set_ylim(self.ylim)
        
        ax[0].axis('equal')

        ax[0].set_title('Structure {}: Energy = {:7.2f}'.format(index, self.energies[index]))

        # Histogram
        if self.first_call:
            self.analysis.plot_histogram(ax[1], self.histogram_kwargs)
            limits = ax[1].get_ylim()
            self.hist_line, = ax[1].plot([self.energies[index], self.energies[index]], [0, limits[1]], color='black')
        else:
            self.hist_line.set_xdata([self.energies[index], self.energies[index]])

        #self.first_call = False

class InteractiveSuccessStats:

    def __init__(self, analysis):

        self.analysis = analysis

        self.CDF = self.analysis.CDF # Convenience/Lazyness
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        self.label_fontsize = self.analysis.label_fontsize
        self.tick_sizes = self.analysis.tick_sizes

        self.where_step = 'post'
        self.figsize = (6, 6)

        # Widgets:
        self.selector = widgets.SelectMultiple(options=self.analysis.labels, value=self.analysis.labels, description='Datasets')
        self.xmin = widgets.BoundedIntText(value=0, min=0, max=self.analysis.max_num_iterations, step=1, description='Min iterations:')
        self.xmax = widgets.BoundedIntText(value=self.analysis.max_num_iterations, min=0, max=self.analysis.max_num_iterations, step=1, description='Max iterations:')

        self.uncertainty_button = widgets.ToggleButtons(options={'On':True, 'Off':False}, 
                description='Uncertainty', 
                button_style='', 
                tooltips=['Show uncertainty', 'Hide uncertainty'], 
                disabled=False)

        self.label_button = widgets.ToggleButtons(
                options={'On':True, 'Off':False}, 
                description='Labels', 
                button_style='', 
                tooltips=['Show labels', 'Hide labes'], 
                disabled=False)               

        self.rate_button = widgets.ToggleButtons(
                options={'On':True, 'Off':False}, 
                description='Rate', 
                button_style='', 
                tooltips=['Show rate', 'Hide rate'], 
                disabled=False, 
                value=False)

        print('???')

        #self.rate_button = widgets.ToggleButton(value=False, description='Ep. for rate', button_style='', tooltip='Control whether or not iteration for rate is plotted', icon='check')
        self.rate_input = widgets.BoundedFloatText(value=0.5, min=0, max=1, step=0.01, description='Rate:')

        # Save button:        
        # self.save_button = widgets.ToggleButton(value=False, description='Save figure?', button_style='danger')
        # self.save_name = widgets.Text(value='default', description='Figure save name')

        self.widget_dict = {'labels':self.selector, 'xmin':self.xmin, 'xmax':self.xmax, 'uncertainty':self.uncertainty_button, 
                            'plot_labels':self.label_button, 'plot_iteration_for_rate':self.rate_button, 'rate':self.rate_input}
        self.widget_list = [self.widget_dict[key] for key in self.widget_dict.keys()]

    def plot_succes_curve(self, ax, index, uncertainty=True):
        color = self.colors[index % len(self.colors)]
        where_step = 'post'
        if uncertainty:
            try:
                ax.fill_between(self.CDF[index][0], self.CDF[index][2], self.CDF[index][3], step=where_step, facecolor=color, alpha=0.1)
            except:
                ax.fill_between(self.CDF[index][0], self.CDF[index][2], self.CDF[index][3], facecolor=color, alpha=0.1)
        ax.step(self.CDF[index][0], self.CDF[index][1], where=self.where_step, c=color, label=self.analysis.labels[index])

        i = index
        max_iterations = np.max(self.analysis.iterations[i])
        if not (self.analysis.iterations[i] == max_iterations).all():
            idx = np.argmin(np.abs(
                np.array(self.CDF[i][0]).reshape(-1, 1) - np.array(self.analysis.iterations[i]).reshape(1, -1)
                ), axis=0)
            x = self.CDF[i][0][idx]
            x[x == self.CDF[i][0][-2]] = np.array(self.analysis.iterations[i])[x == self.CDF[i][0][-2]]
            ax.plot(x, self.CDF[i][1][idx], 'x', color=color)


        #max_iterations = np.max(self.analysis.iterations[index])
        #if not (self.analysis.iterations[index] == max_iterations).all():
        #    idx = np.argmin(np.abs(np.array(self.CDF[index][0]).reshape(-1, 1)-np.array(self.analysis.iterations[index]).reshape(1, -1)), axis=0)
        #    x = self.CDF[index][0][idx]
        #    x[x == self.CDF[index][0][-2]] = np.array(self.analysis.iterations[index])[x == self.CDF[index][0][-2]]
        #    ax.plot(x, self.CDF[index][1][idx], 'x', color=color)

    def plot_rate(self, ax, iteration, rate, index):
        color = self.colors[index % len(self.colors)]
        ax.plot([iteration, iteration], [0, rate], '--', color=color)

    def update_plot(self, labels, xmin, xmax, uncertainty=True, plot_labels=True, plot_iteration_for_rate=False, rate=0.5, save=False, save_name=None):
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if plot_iteration_for_rate:
            iteration_for_rate = self.analysis.get_iterations_for_rate(rate)

        for i, label in enumerate(self.analysis.labels):
            if label in labels:
                self.plot_succes_curve(ax, index=i, uncertainty=uncertainty)

                if plot_iteration_for_rate:
                    self.plot_rate(ax, iteration_for_rate[i], rate, i)

        ax.set_xlabel('iterations', fontsize=self.label_fontsize, color=FONTCOLOR)
        ax.set_ylabel('Succes [%]', fontsize=self.label_fontsize,  color=FONTCOLOR)
        ax.set_ylim([0, 1])
        ax.set_xlim([xmin, xmax])
        ax.tick_params(axis='both', which='major', labelsize=self.tick_sizes, labelcolor=FONTCOLOR)
        if plot_labels:
            ax.legend(fontsize=self.label_fontsize, loc='best')

        if save:
            folder = make_save_folder()
            savefig(folder + save_name + '.png', fig)
            self.save_button.value = False

class InteractiveEnergy:

    def __init__(self, analysis):

        self.analysis = analysis

        self.figsize = (6, 6)
        self.label_fontsize = self.analysis.label_fontsize
        self.tick_sizes = self.analysis.tick_sizes
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


        # Widgets:
        self.selector = widgets.SelectMultiple(options=self.analysis.labels, value=self.analysis.labels, description='Datasets')
        self.xmin = widgets.BoundedIntText(value=0, min=0, max=self.analysis.max_num_iterations, step=1, description='Min iterations:')
        self.xmax = widgets.BoundedIntText(value=self.analysis.max_num_iterations, min=0, max=self.analysis.max_num_iterations, step=1, description='Max iterations:')
        self.ymin_offset = widgets.BoundedFloatText(value=1, min=0, max=25, description='Min. energy')
        self.ymax_offset = widgets.BoundedFloatText(value=50, min=0, max=1000, description='Max. energy')
        
        self.mean_button = widgets.ToggleButtons(
            options={'On':True, 'Off':False},
            value=True, 
            description='Mean energy', 
            button_style='', 
            tooltips=['Show mean', 'Hide mean'], 
            icon='check')

        self.limit_button = widgets.ToggleButtons(
            options={'On':True, 'Off':False},
            value=False, 
            description='Energy limits', 
            button_style='', 
            tooltips=['Show limits', 'Hide limits'], 
            icon='check')

        self.rolling_average = widgets.BoundedIntText(value=1, min=1, max=500, description='Rolling average #:')
        
        # # Save button:        
        # self.save_button = widgets.ToggleButton(value=False, description='Save figure?', button_style='danger')
        # self.save_name = widgets.Text(value='default', description='Figure save name')

        # Widget dict/list
        self.widget_dict = {'labels':self.selector, 'xmin':self.xmin, 'xmax':self.xmax, 'ymin':self.ymin_offset, 'ymax':self.ymax_offset,
                            'plot_mean':self.mean_button, 'plot_limit':self.limit_button, 'rolling_average':self.rolling_average}
        self.widget_list = [self.widget_dict[key] for key in self.widget_dict.keys()]

    def update_plot(self, labels, xmin, xmax, ymin, ymax, plot_mean, plot_limit, rolling_average):
        fig, ax = plt.subplots(figsize=self.figsize)

        energies = self.analysis.energies
        iterations = np.arange(energies.shape[-1])

        policy_only = False

        if policy_only:
            iterations = np.arange(5, energies.shape[-1], 5)
            energies = energies[:, :, iterations-1]
            
        for index, label in enumerate(self.analysis.labels):
            
            if label in labels:

                self.plot_best_energy(ax, index)

                if plot_mean:
                    if not policy_only:
                        self.plot_mean_energy(ax, index)
                    self.plot_mean_energy(ax, index, policy=True)
                    
                if plot_limit:

                    if policy_only:
                        self.plot_energy_limits(ax, index, rolling_average, policy=True)
                    else:
                        self.plot_energy_limits(ax, index, rolling_average, policy=False)

        ax.plot([xmin, xmax],[self.analysis.global_best_energy, self.analysis.global_best_energy], '--k')
        ax.legend()
        ax.set_ylim([self.analysis.global_best_energy-ymin, self.analysis.global_best_energy+ymax])
        ax.set_xlim([xmin, xmax])

        ax.set_xlabel('iteration [#]', fontsize=self.label_fontsize)
        ax.set_ylabel('Energy [eV]', fontsize=self.label_fontsize)

    def plot_best_energy(self, ax, index):
        ax.plot(np.mean(self.analysis.best_energies[index, 0:self.analysis.restarts[index], :], axis=0), '-', label=self.analysis.labels[index],
                    color=self.colors[index])
    
    def plot_mean_energy(self, ax, index, policy=False):
        energies = self.analysis.energies
        iterations = np.arange(energies.shape[-1])
        alpha = 0.8
        color = self.colors[index]
        if policy:
            iterations = np.arange(5, energies.shape[-1], 5)
            energies = energies[:, :, iterations-1]
            alpha = 0.8

        arr = np.nanmean(energies[index, 0:self.analysis.restarts[index], :], axis=0)

        if policy:
            ax.plot(iterations, arr, linestyle='-', label='_nolegend_',
                color='black', alpha=0.5)

        ax.plot(iterations, arr, linestyle='-', label='_nolegend_',
                    color=color, alpha=alpha)
    
    def plot_energy_limits(self, ax, index, w, policy=False):
        energies = self.analysis.energies
        iterations = np.arange(energies.shape[-1])

        if policy:
            iterations = np.arange(5, energies.shape[-1], 5)
            energies = energies[:, :, iterations-1]

        upper_limit = np.convolve(np.max(energies[index, 0:self.analysis.restarts[index]], axis=0), np.ones(w), 'same') / w
        lower_limit = np.convolve(np.min(energies[index, 0:self.analysis.restarts[index]], axis=0), np.ones(w), 'same') / w

        ax.fill_between(iterations, lower_limit, upper_limit, color=self.colors[index], alpha=0.4)

from ase import Atoms
def sorted_species(atoms):
    sorted_atoms = Atoms('', cell=atoms.get_cell())

    indices = np.argsort(atoms.get_atomic_numbers())
    for idx in indices:
        sorted_atoms += atoms[idx]

    return sorted_atoms


