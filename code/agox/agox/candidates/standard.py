from agox.candidates.ABC_candidate import CandidateBaseClass
from ase.calculators.singlepoint import SinglePointCalculator as SPC

class StandardCandidate(CandidateBaseClass):
    """
    I only really made this to keep the style, but I actually think the fairly simple base-class handles everything.
    """
    @classmethod
    def from_atoms(cls, template, atoms):
        candidate =  cls(template=template, positions=atoms.positions, numbers=atoms.numbers, cell=atoms.cell, pbc=atoms.pbc)        
        if hasattr(atoms, 'calc'):
            if atoms.calc is not None:
                if 'energy' in atoms.calc.results:
                    if 'forces' in atoms.calc.results:
                        candidate.calc = SPC(candidate, energy=atoms.get_potential_energy(apply_constraint=False), 
                            forces=atoms.get_forces(apply_constraint=False))
                    else:
                        candidate.calc = SPC(candidate, energy=atoms.get_potential_energy(apply_constraint=False))
        return candidate
