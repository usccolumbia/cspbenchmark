'''
pymatgen=2024.3.1

This script evaluates the performance of CSP algorithms by comparing predicted CIF files with ground truth CIF files.

Usage:
    python success_rates.py --pred_folder CSP_predicted --ground_truth_folder ground_truth

Arguments:
    --pred_folder: Path to the folder containing predicted CIF files.
    --ground_truth_folder: Path to the folder containing ground truth CIF files.

File Naming Conventions:
    - Predicted CIF files: {algorithm}_{formula}.cif (e.g., TCSP_SrTiO3.cif)
    - Ground truth CIF files: {formula}.cif or mp-{formula}.cif (e.g., SrTiO3.cif)

Definitions:
    - StructureMatcher success: For the predicted structure, fit == True means it can find out if similar materials already existed in the Materials Project database.
    - Space group match: Space group number matched between predicted structures and the ground truths.
    - Consensus success rate: Both StructureMatcher and space groups need to be matched between predicted structures and the ground truths.
'''
import argparse
import os
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def get_symmetrized(s):
    sga = SpacegroupAnalyzer(s)
    symmetrized_structure = sga.get_symmetrized_structure()
    return symmetrized_structure

def pmd(structure1, structure2):
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    try:
        rms = sm.get_rms_dist(structure1, structure2)[0]
    except TypeError:
        rms = 'None'
    try:
        rms_anonymous = sm.get_rms_anonymous(structure1, structure2)[0]
    except TypeError:
        rms_anonymous = 'None'
    try:
        fit = sm.fit(structure1, structure2)
    except:
        fit = "None"
    return fit, rms, rms_anonymous

def get_spacegroup(s):
    try:
        analyzer = SpacegroupAnalyzer(s, symprec=0.1)
        space_group = analyzer.get_space_group_number()
    except:
        space_group = 'None'
    return space_group

def main(pred_folder, ground_truth_folder):
    succ_rate_data = []
    fail_list = []
    fail_algo = []
    fail_count = 0
    sg_fail_target = []
    sg_fail_pred = []
    fit_fail = []
    fail_rms = []
    both_match = []

    formula_list = []
    fit_list = []
    rms_list = []
    rms_ano_list = []
    pred_sg_list = []
    target_sg_list = []
    sg_match_list = []
    fit_count = 0
    sg_match_count = 0
    total_count = 0 
    both_match_count = 0

    pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.cif')]
    gt_files = [f for f in os.listdir(ground_truth_folder) if f.endswith('.cif')]

    algorithm_name = os.path.basename(pred_folder)
    
    for pred_file in pred_files:
        formula = pred_file.replace(f'{algorithm_name}_', '').replace('.cif', '')
        gt_file_formula = f'{formula}.cif'
        
        gt_file = None
        if gt_file_formula in gt_files:
            gt_file = gt_file_formula
        
        if gt_file:
            path1 = os.path.join(pred_folder, pred_file)
            path2 = os.path.join(ground_truth_folder, gt_file)
            
            try:
                stru1 = Structure.from_file(path1)
                stru2 = Structure.from_file(path2)  
                
                structure1 = get_symmetrized(stru1)
                structure2 = get_symmetrized(stru2)

                # fit, rms
                fit, rms, rms_anonymous = pmd(structure1, structure2)
                
                # space groups
                pred_sg = get_spacegroup(structure1)
                target_sg = get_spacegroup(structure2)
                sg_match = (pred_sg == target_sg)

                formula_list.append(formula)
                fit_list.append(fit)
                rms_list.append(rms)
                rms_ano_list.append(rms_anonymous)
                pred_sg_list.append(pred_sg)
                target_sg_list.append(target_sg)
                sg_match_list.append(sg_match)
                
                if fit == True:
                    fit_count += 1
                if sg_match:
                    sg_match_count += 1
                total_count += 1
                
                if fit and sg_match:
                    both_match.append(formula)
                    both_match_count += 1
                elif fit and not sg_match:
                    print('-------------Fail---------', formula)
                    fail_list.append(formula)
                    fail_algo.append(pred_folder)
                    sg_fail_target.append(target_sg)
                    sg_fail_pred.append(pred_sg)
                    fit_fail.append(fit)
                    fail_rms.append(rms)

                    os.makedirs(f"fail_results/{formula}/", exist_ok=True)
                    fail_count += 1
                    structure1.to(fmt='cif', filename=f"fail_results/{formula}/{formula}_pred.cif")
                    structure2.to(fmt='cif', filename=f"fail_results/{formula}/{formula}_gt.cif")                

            except Exception as e:
                continue

    # additional row for success rates
    formula_list.append("Success_Rate")
    fit_list.append(fit_count / total_count if total_count > 0 else 0)
    rms_list.append(None)
    rms_ano_list.append(None)
    pred_sg_list.append(None)
    target_sg_list.append("Space_Group_Match_Rate")
    sg_match_list.append(sg_match_count / total_count if total_count > 0 else 0)

    df1 = pd.DataFrame({
        "formula": formula_list,
        "fit": fit_list,
        "rms": rms_list,
        "rms_anonymous": rms_ano_list,
        "pred_sg": pred_sg_list,
        "target_sg": target_sg_list,
        "sg_match": sg_match_list
    })

    # Save total results for success rate in the current folder
    df1.to_csv(f'{algorithm_name}_results.csv', index=None)

    # Save fail results in the current folder
    df2 = pd.DataFrame({"formula": fail_list, "algorithm": fail_algo, "target_sg": sg_fail_target, "pred_sg": sg_fail_pred, "fit": fit_fail, "fail_rms": fail_rms})
    df2.to_csv('fail_results.csv', index=None)

    succ_rate = fit_count / total_count if total_count > 0 else 0
    sg_match_rate = sg_match_count / total_count if total_count > 0 else 0
    both_match_rate = both_match_count / total_count if total_count > 0 else 0
    succ_rate_data.append({
        "Algorithm": algorithm_name,
        "Success_Rate": succ_rate,
        "Space_Group_Match_Rate": sg_match_rate,
        "Consensus_Success_Rate": both_match_rate
    })

    df_succ_rate = pd.DataFrame(succ_rate_data)
    df_succ_rate.to_csv('success_rate.csv', index=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CSP algorithms using predicted and ground truth CIF files.")
    parser.add_argument('--pred_folder', type=str, required=True, help='Path to the folder containing predicted CIF files.')
    parser.add_argument('--ground_truth_folder', type=str, required=True, help='Path to the folder containing ground truth CIF files.')

    args = parser.parse_args()

    main(args.pred_folder, args.ground_truth_folder)
