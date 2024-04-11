import numpy as np
from abc import ABC, abstractmethod
from agox.module import Module

all_feature_types = ['global', 'local', 'global_gradient', 'local_gradient']

class DescriptorBaseClass(ABC, Module):

    feature_types = []

    def __init__(self, surname='', **kwargs):
        Module.__init__(self, surname=surname)
        assert np.array([feature_type in all_feature_types for feature_type in self.feature_types]).all(), 'Unknown feature type declared.'


    ##########################################################################################################
    # Create methods - Implemented by classes that inherit from this base-class.
    # Not called directly by methods of other classes. 
    ##########################################################################################################

    def create_global_features(self, atoms):
        """
        Method to implement on child classes that does the calculate of 
        global feature vectors. 

        This method should not (generally) deal with being given a list of 
        Atoms objects. 

        If implemented 'global' can be added to the child class' feature_types.

        Parameters
        ----------
        atoms : Atoms object

        Returns
        --------
        A single global feature.
        """
        pass

    def create_global_feature_gradient(self, atoms):
        """
        Method to implement on child classes that does the calculate of 
        global feature gradients. 

        This method should not (generally) deal with being given a list of 
        Atoms objects. 

        If implemented 'global_gradient' can be added to the child class' feature_types.

        Parameters
        ----------
        atoms : Atoms object

        Returns
        --------
        A single global feature gradient.
        """
        pass

    def create_local_features(self, atoms):
        """
        Method to implement on child classes that does the calculate of 
        local features.

        This method should not (generally) deal with being given a list of 
        Atoms objects. 

        If implemented 'local' can be added to the child class' feature_types.

        Parameters
        ----------
        atoms : Atoms object

        Returns
        --------
        Local features for each atom in Atoms.
        """
        pass

    def create_local_feature_gradient(self, atoms):
        """
        Method to implement on child classes that does the calculate of 
        local feature gradients.

        This method should not (generally) deal with being given a list of 
        Atoms objects. 

        If implemented 'local_gradient' can be added to the child class' feature_types.

        Parameters
        ----------
        atoms : Atoms object

        Returns
        --------
        A single global feature gradient.
        """
        pass

    ##########################################################################################################
    # Get methods - Ones to use in other scripts.
    ##########################################################################################################

    def get_global_features(self, atoms):
        """
        Method to get global features.

        Parameters
        ----------
        atoms : Atoms object or list or Atoms objects.
            Atoms to calculate features for. 

        Returns
        -------
        list
            Global features for the given atoms.
        """
        self.feature_type_check('global')
        if not (type(atoms) == list):
            atoms = [atoms]
        return [self.create_global_features(a) for a in atoms]

    def get_global_feature_gradient(self, atoms):
        """
        Method to get global features.

        Parameters
        ----------
        atoms : Atoms object or list or Atoms objects.
            Atoms to calculate features for. 

        Returns
        -------
        list
            Global feature gradients for the given atoms.
        """
        self.feature_type_check('global_gradient')
        if not (type(atoms) == list):
            atoms = [atoms]
        return [self.create_global_feature_gradient(a) for a in atoms]

    def get_local_features(self, atoms):
        """
        Method to get local features.

        Parameters
        ----------
        atoms : Atoms object or list or Atoms objects.
            Atoms to calculate features for. 

        Returns
        -------
        list
            Local features for the given atoms.
        """
        self.feature_type_check('local')
        if not (type(atoms) == list):
            atoms = [atoms]
        return [self.create_local_features(a) for a in atoms]

    def get_local_feature_gradient(self, atoms):
        """
        Method to get local feature gradients.

        Parameters
        ----------
        atoms : Atoms object or list or Atoms objects.
            Atoms to calculate features for. 

        Returns
        -------
        list
            Local feature gradients for the given atoms.
        """
        self.feature_type_check('local_gradient')
        if not (type(atoms) == list):
            atoms = [atoms]
        return [self.create_local_feature_gradient(a) for a in atoms]

    def feature_type_check(self, feature_type):
        if not feature_type in self.feature_types:
            raise NotImplementedError(f'This descriptor does not support {feature_type} features')