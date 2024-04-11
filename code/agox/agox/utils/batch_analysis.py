import os
import numpy as np
import matplotlib.pyplot as plt
import re

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from agox.databases.database import Database, export_candidates
from ase.data.colors import jmol_colors
from ase.data import covalent_radii
from datetime import datetime as dt
from argparse import ArgumentParser
import pickle 
from scipy.special import erfinv
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

import warnings

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  ### for disabling tensorflow warnings

from m3gnet.models import M3GNet, Relaxer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

radii_scaling = 30

class Analysis:

    def __init__(self, directories=[], dE=0.05, delta_energy=None, labels=None, save_data=True, force_reload=False, 
                histogram_kwargs = {}):
        self.initial_directories = directories
        self.num_directories = 0 # This is ticked up when a directory is read. 
        self.directories = []
        self.dE = dE
        self.delta_energy = delta_energy
        self.threshold = 0
        self.labels = []
        self.zero_offset = 10000 # Some energy expressions have realistic structures in the positive range, having 0 being the 'failed' energy causes problems for these

        # Plotting settings:
        self.label_fontsize = 12
        self.tick_sizes = 12
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Initial the required lists etc: 
        self.best_energy = np.zeros(self.num_directories)  # Save the best energy for each directoy.
        self.best_idx = []  # Saves the best index for each directory
        self.best_structures = []
        self.names = []
        self.order = []
        self.energies_list = []
        self.best_energies_list = []

        # Saving and loading:
        self.save_data = save_data
        self.force_reload = force_reload

        # Database filtering:
        self.excluded_db = []

        if len(histogram_kwargs) == 0:
            self.histogram_kwargs = {'min_hist_iteration':0, 'max_hist_iteration':0, 'operation':'best', 'threshold':1000}
        else:
            self.histogram_kwargs = histogram_kwargs

        self.xlabel = 'Iteration [#]'

        # energy calculation
        self.aseToPymatgen = AseAtomsAdaptor()
        self.relaxer = Relaxer()
        self.m3gnet_e_form = M3GNet.load()

    def get_feature_image(self, grid):        
        feature = grid.get_feature()
        feature_image = np.zeros([feature.shape[1], feature.shape[0], 3])
        for i in range(feature.shape[1]):
            for j in range(feature.shape[0]):
                for k in range(feature.shape[2]):
                    if feature[j, i, k] == 1:
                        feature_image[i, j] = jmol_colors[grid.atom_types[k]]

        return feature_image

    def add_directory(self, directory, label=None, force_reload=False):
        order, names = [], [] 
        energies, best_energies, best_structure = None, None, None # Placeholders

        # Check that directory isn't already in self.directories:
        if directory in self.directories:
            print('Directory: {} - already been loaded so skipping it'.format(directory))
            return 

        # Information for loading/saving data from this directory:
        cwd = os.getcwd()
        full_path = os.path.join(cwd, directory)
        if full_path[-1] != '/':
            full_path += '/'
        data_path = full_path + 'BA_data/'
        file_names = ['order', 'names', 'energies', 'best_energies', 'best_structure']

        if os.path.exists(data_path) and not force_reload:
            print('Reloading files from {}'.format(data_path))
            loaded_data = []
            for file_name in file_names:
               with open(data_path+file_name+'.pckl', 'rb') as f:
                   loaded_data.append(pickle.load(f))
            order, names, energies, best_energies, best_structure = loaded_data
            # print(energies)
            # print(best_energies)
            # print(order)
            # print(names)
            
        else:
            # Find database files in this directory:
            dbs = []            
            print('Loading files from directory: {}'.format(directory))
            for f in os.listdir(os.path.join(cwd, directory)):
                if f[-3:] == '.db':
                    order.append(int(re.findall('\d+', f)[0]))  # noqa
                    names.append(f)
                    if os.stat(os.path.join(cwd, directory, f)).st_size < 5000:
                        self.excluded_db.append(order[-1])
 
            # Filter database files - Check sizes etc
            #for excluded in self.excluded_db:
            #    print(excluded)
            #    name = 'db{}.db'.format(excluded)
            #    names.remove(name)
            #    order.remove(excluded)

            sort_idx = np.argsort(order).astype(int)
            order = np.array(order)[sort_idx]
            names = np.array(names)[sort_idx]

            # Gather data from databases
            for name in names:
                database = Database(os.path.join(cwd, directory, name))
                dbs.append(database)

            energies, best_energies, best_structure = export_candidates(dbs)


            # # Get max energy
            # max_E =-1000
            # for energy_list in energies:
            #     for e in energy_list:
            #         if not np.isnan(e):
            #             if e>max_E:
            #                 max_E = e

            # self.nan_energy = max_E + 1

            # # Replace nan values with self.nan_energy
            # for energy_list in energies:
            #     for i,e in enumerate(energy_list):
            #         if np.isnan(e):
            #             energy_list[i] = self.nan_energy

            # for energy_list in best_energies:
            #     for i,e in enumerate(energy_list):
            #         if np.isnan(e):
            #             energy_list[i] = self.nan_energy                        


            # Save some stuff:
            if self.save_data:
                if not os.path.exists(data_path):
                    os.mkdir(data_path)
                
                things = [order, names, energies, best_energies, best_structure]
                for thing, file_name in zip(things, file_names):
                    with open(data_path+file_name + '.pckl', 'wb') as f:
                        pickle.dump(thing, f)

                
        # Append information to lists
        self.names.append(names)
        self.order.append(order)
        self.energies_list.append(energies)
        self.best_energies_list.append(best_energies)
        self.best_structures.append(best_structure)

        # When succesfully loaded the directory can be added to the internal count.
        self.directories.append(directory)
        self.num_directories += 1

        # Added to list of plot-labels: 
        if label is None:
            idx = directory.rfind('/')

            if idx == len(directory) - 1:
                idx = directory[0:-1].rfind('/')
                self.labels.append(directory[idx + 1:-1])
            else:
                self.labels.append(directory[idx + 1::])
        else:
            self.labels.append(label)

    def compile_information(self):
        """
        Gather information. 
        """
        self.iterations = []
        self.restarts = np.zeros(self.num_directories).astype(int)
        for i in range(self.num_directories):
            # self.iterations[i] = np.max([len(arr) for arr in energies_list[i]])
            self.iterations.append([len(arr) for arr in self.energies_list[i]])
            self.restarts[i] = len(self.energies_list[i])
            
        self.max_num_restarts = np.max(self.restarts).astype(int)
        self.max_num_iterations = np.max([np.max(iterations) for iterations in self.iterations]).astype(int)

        self.energies = np.zeros((self.num_directories, self.max_num_restarts, self.max_num_iterations)) + self.zero_offset
        self.best_energies = np.zeros((self.num_directories, self.max_num_restarts, self.max_num_iterations)) + self.zero_offset

        failed_to_load = []
        for i in range(self.num_directories):
            for j, dat1, dat2 in zip(np.arange(len(self.energies_list[i])), self.energies_list[i], self.best_energies_list[i]):
                try:
                    self.energies[i, j, 0:len(dat1)] = dat1
                    self.best_energies[i, j, 0:len(dat2)] = dat2
                    self.best_energies[i, j, len(dat2)::] = np.nanmin(dat2)
                except ValueError:
                    failed_to_load.append([i, j])


        self.best_energy = np.nanmin(np.nanmin(self.best_energies, axis=2), axis=1)
        self.global_best_energy = np.nanmin(self.best_energies)
        self.global_best_structure = self.best_structures[np.nanargmin(self.best_energy)]

        self.num_atoms = len(self.global_best_structure)

        # Threshold
        #self.energies[self.energies > self.threshold] = self.threshold
        #self.best_energies[self.best_energies > self.threshold] = self.threshold

        # Figure out which iteration in each run produced the best iteration:
        safe_best_energies = self.best_energies.copy()
        safe_best_energies[np.isnan(self.best_energies)] = 10E10
        self.best_indicies = np.argmin(safe_best_energies, axis=-1)

    def analyse_directories(self):
        for idx, directory in enumerate(self.initial_directories):

            self.add_directory(directory, force_reload=self.force_reload)

        self.compile_information()

    def analyse_directories_old(self):        
        #self.best_energy = np.zeros(self.num_directories)  # Save the best energy for each directoy.
        #self.best_idx = []  # Saves the best index for each directory
        #self.best_structures = []
        #self.names = []
        #self.order = []

        energies_list = []
        best_energies_list = []

        t0 = dt()
        for idx, directory in enumerate(self.directories):
            print('Loading files from directory: {}'.format(directory))

            dbs = []
            cwd = os.getcwd()

            order = []
            names = []
            for f in os.listdir(os.path.join(cwd, directory)):
                if f[-3:] == '.db':
                    order.append(int(re.findall('\d+', f)[0]))  # noqa
                    names.append(f)
            sort_idx = np.argsort(order).astype(int)

            order = np.array(order)[sort_idx]
            names = np.array(names)[sort_idx]

            for name in names:
                database = Database(os.path.join(cwd, directory, name))
                dbs.append(database)

            # Gather data from databases
            energies, best_energies, best_structure = export_candidates(dbs)

            energies_list.append(energies)
            best_energies_list.append(best_energies)
            self.names.append(names)
            self.order.append(order)


        t1 = dt()

        # self.iterations = np.zeros(self.num_directories).astype(int)
        self.iterations = []
        self.restarts = np.zeros(self.num_directories).astype(int)
        for i in range(self.num_directories):
            # self.iterations[i] = np.max([len(arr) for arr in energies_list[i]])
            self.iterations.append([len(arr) for arr in energies_list[i]])
            self.restarts[i] = len(energies_list[i])

        self.max_num_restarts = np.max(self.restarts).astype(int)
        self.max_num_iterations = np.max(np.max([iterations for iterations in self.iterations])).astype(int)

        self.energies = np.zeros((self.num_directories, self.max_num_restarts, self.max_num_iterations))
        self.best_energies = np.zeros((self.num_directories, self.max_num_restarts, self.max_num_iterations))

        for i in range(self.num_directories):
            for j, dat1, dat2 in zip(np.arange(len(energies_list[i])), energies_list[i], best_energies_list[i]):
                self.energies[i, j, 0:len(dat1)] = dat1
                # self.energies[i, j, len(dat1)::] = np.nan
                self.best_energies[i, j, 0:len(dat2)] = dat2
                self.best_energies[i, j, len(dat2)::] = np.nanmin(dat2)
                
                # self.best_energies[i, j, len(dat2)::] = np.nan

        self.best_energy = np.min(np.min(self.best_energies, axis=2), axis=1)
        self.global_best_energy = np.min(self.best_energies)
        self.global_best_structure = best_structure  # self.best_structures[np.argmin(self.best_energy)]

        self.num_atoms = len(best_structure)

        t2 = dt()

        print(f'Loading files took {t1 - t0} s')
        print(f'Processing took {t2 - t1} s')

        self.energies[self.energies > self.threshold] = self.threshold
        self.best_energies[self.best_energies > self.threshold] = self.threshold

        # Figure out which iteration in each run produced the best iteration:
        self.best_indicies = np.argmin(self.best_energies, axis=-1)

    def calculate_CDF(self):

        if self.delta_energy is None:
            delta_energy = self.dE * self.num_atoms
        else:
            self.dE = self.delta_energy / self.num_atoms
            delta_energy = self.delta_energy
        
        print('Using dE of {:6.4f} eV pr. atom ({} atoms)'.format(self.dE, self.num_atoms))

        best_energies = self.best_energies.copy()
        best_energies[np.isnan(best_energies)] = self.global_best_energy + 1E5

        booled = best_energies <= self.global_best_energy + delta_energy

        events_np = np.max(booled, axis=2).astype(int)
        times_np = np.argmax(booled, axis=2)
        #times_np[(times_np == 0) * (events_np == 0)] = self.max_num_iterations - 1
        times_np[(events_np == 0)] = self.max_num_iterations - 1
        times_np[(events_np == 1) * (times_np == 0)] = 1
        self.CDF = self.get_stats(times_np, events_np)

        # Determine when the minimum was found:
        self.succes = booled[:, :, -1]

    def get_stats(self, all_times, all_events):

        CDF = []

        for dir_idx in range(len(all_times)):            
            times = all_times[dir_idx]
            events= all_events[dir_idx]

            sort_index = np.argsort(times)
            events_sorted = events[sort_index]
            times_sorted = times[sort_index]

            #N_total = len(events) # All restarts are either dead or censored at the end. 
            N_total = self.restarts[dir_idx]
            ntimes = len(np.unique(times))

            n = np.ones(ntimes) * N_total
            d = np.zeros(ntimes)

            # Precalculate some things:
            for i, time in enumerate(np.unique(times_sorted)):
                d[i] = np.sum(events_sorted[times_sorted == time]) # Runs that succeed at 'time'
                n[i+1:] -= d[i] # NUmber of runs yet to succeed is reduced. 

            # Caculate KM: 
            km = np.ones(ntimes+1)
            times_axis = np.concatenate([[0], np.unique(times_sorted)])
            for i in range(1, len(km)):
                km[i] = km[i-1] * (1-(d[i-1]/n[i-1]))

            # Uncertainties: Small numbers are for error avoidance, they dont change anything 
            # unless you look at digits you shouldnt be looking at.
            V = 1 / (np.log(km + 0.00000001) ** 2 + 0.000001)
            t = 0

            for i in range(1, len(km)):
                if n[i-1] - d[i-1] != 0:
                    t += d[i-1] / (n[i-1] * (n[i-1] - d[i-1]))
                else:
                    t = 0
                V[i] *= t
            V = np.nan_to_num(V, 0)

            alpha = 0.95
            z = np.sqrt(2)*erfinv(2*(alpha+(1-alpha)/2.)-1)

            # print('???')
            # stabilizer = 1E-10 # To supress useless errors.
            # a = np.log(km)            
            # a = np.abs(-a)
            # a = np.log(a)
            # a = a + z * np.sqrt(V)
            # a = np.exp(a)
            # a = np.exp(-a)
            # print('???')
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cu = np.exp(-np.exp(np.log(-np.log(km)) - z*np.sqrt(V)))
                cl = np.exp(-np.exp(np.log(-np.log(km)) + z*np.sqrt(V)))

            CDF.append([times_axis, 1-km, 1-cl, 1-cu])

        return CDF

    def plot_CDF(self, ax, factor=1):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(self.num_directories):
            color = colors[i % len(colors)]
            ax.fill_between(self.CDF[i][0], self.CDF[i][2]*factor, self.CDF[i][3]*factor, step='post', facecolor=color, alpha=0.1)
            ax.step(self.CDF[i][0], self.CDF[i][1]*factor, where='post', c=color, label=self.labels[i])
            max_iterations = np.max(self.iterations[i])
            if not (self.iterations[i] == max_iterations).all():
                idx = np.argmin(np.abs(
                    np.array(self.CDF[i][0]).reshape(-1, 1) - np.array(self.iterations[i]).reshape(1, -1)
                    ), axis=0)
                x = self.CDF[i][0][idx]
                x[x == self.CDF[i][0][-2]] = np.array(self.iterations[i])[x == self.CDF[i][0][-2]]
                ax.plot(x, self.CDF[i][1][idx]*factor, 'x', color=color)

            # ax.plot(
            #         self.CDF[i][0][self.censoring[i]], self.CDF[i][1][self.censoring[i]], 
            #         ls='None', marker='+', c=color
            #         )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel('Success [%]')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, self.max_num_iterations])
        ax.legend()

    def plot_histogram(self, ax, histogram_kwargs=None):

        # Get settings from dict:
        if histogram_kwargs is None:
            min_hist_iteration = self.histogram_kwargs['min_hist_iteration']
            max_hist_iteration = self.histogram_kwargs['max_hist_iteration']
            operation = self.histogram_kwargs['operation']
            threshold = self.histogram_kwargs['threshold']
        else:
            min_hist_iteration = histogram_kwargs['min_hist_iteration']
            max_hist_iteration = histogram_kwargs['max_hist_iteration']
            operation = histogram_kwargs['operation']
            threshold = histogram_kwargs['threshold']

        # Clean-up inputs:
        if not operation in ['best', 'all']:
            print("WARNING: Operation {} not implemented - Defaulting to 'best'")
            operation = 'best'

        if max_hist_iteration == 0:
            max_hist_iteration = self.energies.shape[2]-1

        if threshold != 0:
            threshold = self.global_best_energy + threshold 

        # Apply 'operation':
        if operation == 'best': # Histogram over the energies of the BEST structures pr. restart (so min over iterations dimension effectively)
            data = self.best_energies[:, :, max_hist_iteration-1]
        elif operation == 'all': # Histogram over the energies of ALL structures pr. iterations within the iteration limits.
            data = self.energies[:, :, min_hist_iteration:max_hist_iteration]
        
        # Define histogram range:
        min_val = np.nanmin(data[:, :])
        max_val = np.nanmax(data[:, :][data[:, :]<threshold])

        width = self.dE * self.num_atoms

        #num_buckets = abs(int((max_val - min_val) // width))+2
        num_buckets = (np.ceil((max_val - min_val) / width) + 1).astype(int)
        buckets = [min_val+(i*width) for i in range(num_buckets)]

        if buckets == 0:
            buckets = 1

        ymax = 0
        histograms = []; lines = []
        for dir_idx in range(self.num_directories):
            vals = data[dir_idx, :]
            hist, bin_edges = np.histogram(vals, bins=buckets, range=(min_val, max_val))

            hist = hist.astype(np.float64)
            hist *= 100 / self.restarts[dir_idx]
            patches = []
            for j in range(0, len(bin_edges) - 1):
                p = Rectangle((bin_edges[j], 0), width, hist[j], fill=True)
                patches.append(p)

            pc = PatchCollection(patches, alpha=0.7, facecolor=self.colors[dir_idx], edgecolor='black')
            ax.add_collection(pc)

            ymax = np.max([ymax, np.max(hist)])

            lines.append(plt.plot([-10, -10], [-10, -5], color=self.colors[dir_idx], label=self.labels[dir_idx], linewidth=2))

            histograms.append((hist, bin_edges))


        ax.set_xlim([min_val - width * 2, max_val + width * 2])
        ax.set_ylim([0, ymax + 5])
        ax.plot([0, 0], [0, 0], color='white')
        ax.set_xlabel('Energy [eV]', fontsize=self.label_fontsize)
        ax.set_ylabel('Count [%]', fontsize=self.label_fontsize)

        ax.legend()
        return histograms, lines

    def plot_structure(self, ax, structure):
        image = self.get_feature_image(structure)
        ax.imshow(image, cmap='gray', interpolation='nearest')
        ax.set_title('Best Structure')
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_structure_nice(self, ax, structure, scaling=radii_scaling, energy=None):
        cell = structure.get_cell()
        positions = structure.get_positions()

        # Sort according to Z-axis: 
        sort_idx = np.argsort(positions[:, 2])
        positions = positions[sort_idx, :]

        atom_types = structure.get_atomic_numbers()[sort_idx]
        colors = [jmol_colors[num] for num in atom_types]
        sizes = np.power(np.array([covalent_radii[num] for num in atom_types]) * scaling, 2)
        scat = ax.scatter(positions[:, 0], positions[:, 1], color=colors, edgecolor='black', s=sizes)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        #ax.set_xlim([0, cell[0, 0]])
        #ax.set_ylim([0, cell[1, 1]])

    def plot_energy(self, ax, set_labels=True):
        ax.set_title('Energy')
        for dir_idx in range(self.num_directories):
            color = self.colors[dir_idx % len(self.colors)]

            mean = np.nanmean(self.energies[dir_idx, 0:self.restarts[dir_idx], :], axis=0)
            
            ax.plot(
                    np.nanmean(self.energies[dir_idx, 0:self.restarts[dir_idx], :], axis=0), '-', color=color,
                    label=self.labels[dir_idx], alpha=0.5)
            ax.plot(
                    np.nanmean(self.best_energies[dir_idx, 0:self.restarts[dir_idx], :], axis=0), '-', label='_nolegend_',
                    color=color
                    )

        ax.legend()
        if set_labels:
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel('Energy [eV]')

    def print_info(self, print_all):
        for num, directory in enumerate(self.directories):
            finished = (self.iterations[num] == np.max(self.iterations[num])).all()
            print('=' * 10 + ' ' + directory + ' ' + '=' * 10)
            print(f'Finished: {finished}'.format(finished))
            print(f'Number of restarts: {self.restarts[num]}')
            print(f'Number of iterations: {np.max(self.iterations[num])}')
            print(f'Success rate: {np.nanmax(self.CDF[num][1]) * 100:6.3f} %')
            print(f'Best energy before relaxing: {np.nanmin(self.best_energies[num, :, :]):.5f}\n')

            if print_all:
                order_list = []
                for restart in range(self.restarts[num]):
                    print(
                            f'File: {self.names[num][restart]} - GM found: {self.succes[num, restart]}'
                            f'- Best Energy: {np.min(self.best_energies[num, restart, :])}'
                            )
                    if self.succes[num, restart]:
                        order_list.append(self.order[num][restart])
                # Print the numbers of files that succeeded:
                # print(str(order_list))

    def get_best_structures(self, directory=None):
        structures = []
        cwd = os.getcwd()
        energies = []

        result_file = open(self.directories[0] + "result.txt", "w")

        if directory is None:
            directory = np.arange(len(self.directories))

        for i, directory in enumerate([self.directories[i] for i in directory]):
            for j, name in enumerate(self.names[i]):
                try:
                    database_path =  os.path.join(cwd, directory, name)
                    #structures.append(get_best(database_path))
                    structures.append(Database(database_path).get_best_structure())
                    # print(len(structures))
                    structures[-1].db = os.path.join(directory, name)
                    structure_pymatgen = self.aseToPymatgen.get_structure(structures[-1])
                    energy = self.m3gnet_e_form.predict_structure(structure_pymatgen)
                    energy = energy[0][0].numpy().astype(float) / len(structure_pymatgen)
                    # energy = structures[-1].get_potential_energy()
                    energies.append(energy)
                except Exception as e:
                    pass

        indexs = np.argsort(energies).astype(int)
        structures = [structures[i] for i in indexs]
        result_file.write(f'best energy before relaxing: {min(energies)}\n\n')

        ### best structure (default)
        print(f'ase structure:\n-------------------\n {structures[0]}\n')
        result_file.write(f'ase structure:\n-------------------\n {structures[0]}\n\n')
        structure_pymatgen = self.aseToPymatgen.get_structure(structures[0])
        print(f'pymatgen structure:\n--------------------\n {structure_pymatgen}\n')
        result_file.write(f'pymatgen structure:\n--------------------\n {structure_pymatgen}\n\n')

        ### best structure (symmetrized)
        analyzer = SpacegroupAnalyzer(structure_pymatgen, symprec = 0.01)
        structure_pymatgen = analyzer.get_symmetrized_structure()
        print(f'symmetrized pymatgen structure:\n---------------------------------\n {structure_pymatgen}\n')
        result_file.write(f'symmetrized pymatgen structure:\n---------------------------------\n {structure_pymatgen}\n\n')

        ### best structure (relaxed)
        nAtoms = len(structure_pymatgen)
        structure_pymatgen = self.relaxer.relax(structure_pymatgen)
        relaxed_structure_energy = float(structure_pymatgen['trajectory'].energies[-1] / nAtoms)
        structure_pymatgen = structure_pymatgen["final_structure"]
        print(f'relaxed pymatgen structure:\n----------------------------\n {structure_pymatgen}\n')
        result_file.write(f'relaxed pymatgen structure:\n----------------------------\n {structure_pymatgen}\n\n')
        print(f'final energy: {relaxed_structure_energy}')
        result_file.write(f'final energy: {relaxed_structure_energy}')


        # results

        cifWriter = CifWriter(structure_pymatgen, symprec = 0.01)
        cifWriter.write_file(self.directories[0] + 'relaxed_best_structure.cif')
        print(f'Relaxed structure saved in {self.directories[0]}relaxed_best_structure.cif')
        energies = np.array(energies)[indexs]
        # print(f'energies: {energies}')
        result_file.close()

        return structures, energies

    def animate_structure(self, ax):
        from agox.helpers.plot_confinement import plot_cell

        structures, energies = self.get_best_structures()

        cell = structures[0].get_cell()
        positions = structures[0].get_positions()
        atom_types = structures[0].get_atomic_numbers()

        # Sort according to Z-axis: 
        sort_idx = np.argsort(positions[:, 2])
        positions = positions[sort_idx, :]
        atom_types = atom_types[sort_idx]

        colors = [jmol_colors[num] for num in atom_types]
        sizes = np.power(np.array([covalent_radii[num] for num in atom_types]) * radii_scaling, 2)

        scat = ax.scatter(positions[:, 0], positions[:, 1], color=colors, edgecolor='black', s=sizes)

        plot_cell(ax, cell, np.array([0, 0, 0]), color='black', linestyle='--', use_combinations=False)

        # im = ax.imshow(images[0], cmap='gray', interpolation='nearest')
        ax.set_title('Best Structure')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        #ax.set_xlim([0, cell[0, 0]])
        #ax.set_ylim([0, cell[1, 1]])

        return scat, structures, energies

    def plot_energy_distribution(self, ax, energy_range = 50, iteration_start=100, iteration_end=101, best=True): # New
        """
        Alternative to histogram for visualizing the distribution of energies of the structures ASLA finds. 

        ax - obj: matplotlib axis to plot in. 
        energy_range - int: Span of energy axis
        iteration_start - int: Starting iteration to consider.
        iteration_end - int: Ending iteration to consider. 
        best: Consider the best energy for each restart. 

        To do:

        Animation: Would be nice if the structure plot points out the current energy structure, similarly to the histogram - although a way 
                   to handle conflicts between the two needs to be figured out. 

        Parameters: The parameters are pretty arbitary at the moment, it would be nice if the width of the gaussian was related to dE in some 
                    easily understandable way. The amplitude parameter defaults to 1 so that the y-axis can be interpreted as counts, but not sure 
                    how to handle a conversion to percentage.
                    With the correct ratio between the amplitude and the width the integral could num_restart * num_iterations, at which point a 
                    percentage interpretation might be meaningful (or even useful?). 
        """

        def gaussian(E, Ei, A=1, w=1):
            Ei = Ei.reshape(-1, 1) 
            return np.sum(A * np.exp(-(E-Ei)**2 / w**2), axis = 0)

        # Determine the energy scale:
        Emin = self.global_best_energy - 10 #(self.dE * self.num_atoms) * 2
        Emax = self.global_best_energy + energy_range
        num_steps = (Emax - Emin) / (self.dE)

        E = np.linspace(Emin, Emax, num_steps).reshape(1, -1)
        D = np.zeros((self.num_directories, int(num_steps)))


        for d in range(self.num_directories):
            if not best:
                D[d] += gaussian(E, self.energies[d, 0:self.restarts[d], iteration_start:iteration_end])
            else:
                D[d] += gaussian(E, self.best_energies[d, 0:self.restarts[d], -1])

        # Plotting
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for d in range(self.num_directories):
            ax.plot(E[0, :], D[d, :], color=colors[d])
            ax.fill_between(E[0, :], np.zeros_like(D[d, :]), D[d, :], color=colors[d], alpha=0.25)

        ylim = ax.get_ylim()
        #ax.plot([self.global_best_energy, self.global_best_energy], [ylim[0], ylim[1]], color='red', linestyle='--')
        #ax.plot([self.global_best_energy+(self.dE*self.num_atoms), self.global_best_energy+(self.dE*self.num_atoms)], [ylim[0], ylim[1]], color='red', linestyle='--')
        ax.set_ylim([0, ylim[1]])
        ax.set_xlim([Emin, Emax])
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('Counts [#]')
        return D, E

    def get_iterations_for_rate(self, rate): # New
        CDF = np.array(self.CDF)
        iteration_for_rate = np.zeros(len(CDF))
        for j in range(len(CDF)):
            index = np.argmin(np.abs(CDF[j, 1]-rate))
            iteration_for_rate[j] = CDF[j, 0][index]

        return iteration_for_rate

####################################################################################################################
# Interactive: Functions for matplotlib interactivity
####################################################################################################################

class KeyBoardEvent:

    def __init__(self, struct_scat, structures, energies, named_axes):
        self.ind = 0

        self.structures = structures
        self.energies = energies
        self.num = len(self.structures)

        # Plot objects:
        self.struct_scat = struct_scat

        # Axes:
        self.named_axes = named_axes
        self.hist_ax = self.named_axes[3]

        self.hist = self.hist_ax is not None

        if self.hist:
            limits = self.hist_ax.get_ylim()
            self.hist_line, = self.hist_ax.plot([self.energies[0], self.energies[0]], [0, limits[1]], color='black')

    def update_scatter(self, index):
        cell = self.structures[index].get_cell()
        structure = self.structures[index].repeat((3,3,1))
        structure.translate(-cell[0]-cell[1])
        positions = structure.get_positions()
        atomic_numbers = structure.get_atomic_numbers()

        # Sort according to z-coordiantes   
        sort_idx = np.argsort(positions[:, 2])
        positions = positions[sort_idx, :][:, 0:2]
        atomic_numbers = atomic_numbers[sort_idx]

        colors = np.array([jmol_colors[t] for t in atomic_numbers])
        sizes = np.power(np.array([covalent_radii[num] for num in atomic_numbers]) * radii_scaling, 2)

        self.struct_scat.set_offsets(positions)
        self.struct_scat.set_facecolor(colors)
        self.struct_scat.set_sizes(sizes)
        try:
            tempStruct = self.aseToPymatgen.get_structure(self.structures[index])
            energy = self.m3gnet_e_form.predict_structure(tempStruct)
            energy = energy[0][0].numpy().astype(float) / len(tempStruct)
            # energy = '${:8.3f}$'.format(self.structures[index].get_potential_energy())
            energy = '${:8.3f}$'.format(energy)
        except:
            energy = 'NaN'
        title = '{}\n{}'.format(self.structures[index].db,energy)
        self.named_axes[0].set_title(title)

    def update_hist(self, index):
        self.hist_line.set_xdata([self.energies[index], self.energies[index]])

    def press(self, event):
        if event.key == 'right':
            self.ind += 1
            i = self.ind % self.num
            self.update_scatter(i)

            if self.hist:
                self.update_hist(i)

            fig.canvas.draw_idle()

        if event.key == 'left':
            self.ind -= 1
            i = self.ind % self.num
            self.update_scatter(i)

            if self.hist:
                self.update_hist(i)

            fig.canvas.draw_idle()

def command_line_analysis():
    # Input arguments:
    parser = ArgumentParser()
    parser.add_argument('-d', '--directories', nargs='+', type=str)  # List - Directories
    parser.add_argument('-dE', '--delta_energy', type=float, default=0.05)
    parser.add_argument('-DE', '--Delta_energy', type=float, default=None)
    parser.add_argument('--save_name', type=str, default=None)

    # Flags
    parser.add_argument('-c', '--plot_cdf', action='store_true')         # Bool - Plot CDF    
    parser.add_argument('-s', '--plot_structure', action='store_true')   # Bool - Plot structure
    parser.add_argument('-e', '--plot_energies', action='store_true')    # Bool - Plot energy
    parser.add_argument('-hg', '--plot_histogram', action='store_true')   # Bool - Plot histogram
    parser.add_argument('-p', '--print_all', action='store_true')        # Print everything
    parser.add_argument('-v', '--view_best', action='store_true')
    parser.add_argument('-g', '--save', action='store_true')             # Save the figure?
    parser.add_argument('-sd', '--save_data', action='store_false')
    parser.add_argument('-fr', '--force_reload', action='store_true')

    # Histogram settings:
    parser.add_argument('-hgl', '--histogram_limits', nargs='+', default=[0, 0]) # Histogram limits: min iteration, max iteration
    parser.add_argument('-hgop', '--histogram_operation', type=str, default='best') # Histogram operations: best, all
    parser.add_argument('-hgth', '--histogram_threshold', type=int, default=1000) # Histogram energy threshold: 0 gives all (negative) energies, positive number gives that over the best found.

    # Animation related things:
    parser.add_argument('-a', '--animate', action='store_false')        # Print everything

    args = parser.parse_args()
    directories = args.directories

    if args.Delta_energy is None:
        dE = args.delta_energy
        delta_energy = None
    else:
        dE = None
        delta_energy = args.Delta_energy

    # Plots:
    plot_cdf = args.plot_cdf
    plot_structure = args.plot_structure
    plot_energies = args.plot_energies
    plot_histogram = args.plot_histogram
    num_plots = plot_cdf + plot_structure + plot_energies + plot_histogram
    print_all = args.print_all
    view_best = args.view_best

    histogram_kwargs = {
                        'min_hist_iteration':int(args.histogram_limits[0]), 
                        'max_hist_iteration':int(args.histogram_limits[1]),
                        'operation':args.histogram_operation,
                        'threshold':args.histogram_threshold
                        }

    # Save/reload
    save_data = args.save_data
    force_reload = args.force_reload

    # Saving?
    save = args.save
    save_name = args.save_name

    animate = args.animate

    # Start
    A = Analysis(directories, dE=dE, save_data=save_data, force_reload=force_reload, histogram_kwargs=histogram_kwargs, delta_energy=delta_energy)
    #A = Analysis(directories, dE=dE, save_data=True, force_reload=False)
    
    A.analyse_directories()
    A.calculate_CDF()

    A.print_info(print_all)

    # Open the best structure in the ASE GUI.
    if view_best:
        A.global_best_structure.edit()

    if num_plots > 0:
        fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 5, 5))
        if num_plots > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        ax_idx = 0
        if plot_structure:
            if animate:
                struct_scat, structures, energies = A.animate_structure(axes[ax_idx])
            else:
                A.plot_structure(axes[ax_idx], A.global_best_structure)

            struct_ax = axes[ax_idx]
            ax_idx += 1
        else:
            struct_ax = None

        if plot_cdf:
            A.plot_CDF(axes[ax_idx])
            cdf_ax = axes[ax_idx]
            ax_idx += 1
        else:
            cdf_ax = None

        if plot_energies:
            A.plot_energy(axes[ax_idx])
            energy_ax = axes[ax_idx]
            ax_idx += 1
        else:
            energy_ax = None

        if plot_histogram:
            A.plot_histogram(axes[ax_idx])
            hist_ax = axes[ax_idx]
            ax_idx += 1
        else:
            hist_ax = None

        if animate:
            if plot_structure:
                named_axes = [struct_ax, cdf_ax, energy_ax, hist_ax]
                callback = KeyBoardEvent(struct_scat, structures, energies, named_axes=named_axes)
                fig.canvas.mpl_connect('key_press_event', callback.press)
            else:
                print("To use the animation, the structure has to be plotted")

        plt.tight_layout()

        if save:
            if save_name is None:
                name_preifx = directories[0].split('/')
                if name_preifx[-1] != '':
                    name_preifx = name_preifx[-1]
                else:
                    name_preifx = name_preifx[-2]
                save_name = name_preifx + '_best_structure'
            
            fig_path_prefix = directories[0] + 'figures/'
            if not os.path.exists(fig_path_prefix):
                os.mkdir(fig_path_prefix)
            plt.savefig(os.path.join(fig_path_prefix + save_name + dt.now().strftime('%Y_%m_%d_%H_%S') + '.png'), bbox_inches='tight', dpi=300)
        plt.show()

if __name__ == '__main__':
    command_line_analysis()