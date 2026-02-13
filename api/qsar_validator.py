#!/usr/bin/env python3
"""
QSAR Results Validation Tool

This script helps validate QSAR predictions against external databases and tools.
It provides multiple validation methods for scientific credibility.
"""

import requests
import json
import time
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
import os

class QSARValidator:
    def __init__(self):
        self.validation_results = {}
        
    def validate_compound_properties(self, compound_name, smiles, your_predictions):
        """
        Validate compound properties against multiple external sources
        
        Args:
            compound_name (str): Name of the compound
            smiles (str): SMILES string of the compound
            your_predictions (dict): Your QSAR predictions to validate
        """
        print(f"\n🔬 Validating: {compound_name}")
        print(f"SMILES: {smiles}")
        print(f"Your predictions: {your_predictions}")
        
        results = {
            'compound': compound_name,
            'smiles': smiles,
            'your_predictions': your_predictions,
            'external_validation': {}
        }
        
        # 1. SwissADME validation (free web tool)
        swiss_results = self.validate_with_swissadme(smiles)
        if swiss_results:
            results['external_validation']['SwissADME'] = swiss_results
            
        # 2. RDKit calculated descriptors (open source)
        rdkit_results = self.calculate_rdkit_descriptors(smiles)
        if rdkit_results:
            results['external_validation']['RDKit'] = rdkit_results
            
        # 3. PubChem data lookup
        pubchem_results = self.get_pubchem_data(compound_name)
        if pubchem_results:
            results['external_validation']['PubChem'] = pubchem_results
            
        # 4. ChEMBL bioactivity data
        chembl_results = self.get_chembl_data(compound_name)
        if chembl_results:
            results['external_validation']['ChEMBL'] = chembl_results
            
        self.validation_results[compound_name] = results
        return results
    
    def calculate_rdkit_descriptors(self, smiles):
        """Calculate molecular descriptors using RDKit (open source validation)"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            descriptors = {
                'molecular_weight': round(Descriptors.MolWt(mol), 2),
                'logp': round(Descriptors.MolLogP(mol), 3),
                'tpsa': round(rdMolDescriptors.CalcTPSA(mol), 2),
                'hbd': rdMolDescriptors.CalcNumHBD(mol),
                'hba': rdMolDescriptors.CalcNumHBA(mol),
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'source': 'RDKit (Open Source)'
            }
            
            print(f"  ✅ RDKit descriptors calculated")
            return descriptors
            
        except Exception as e:
            print(f"  ❌ RDKit calculation failed: {e}")
            return None
    
    def get_pubchem_data(self, compound_name):
        """Get compound data from PubChem database"""
        try:
            # Search for compound by name
            search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/property/MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount/JSON"
            
            response = requests.get(search_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                    props = data['PropertyTable']['Properties'][0]
                    
                    pubchem_data = {
                        'molecular_weight': props.get('MolecularWeight'),
                        'logp': props.get('XLogP'),
                        'tpsa': props.get('TPSA'),
                        'hbd': props.get('HBondDonorCount'),
                        'hba': props.get('HBondAcceptorCount'),
                        'source': 'PubChem Database'
                    }
                    
                    print(f"  ✅ PubChem data retrieved")
                    return pubchem_data
                    
            print(f"  ⚠️ PubChem: No data found for {compound_name}")
            return None
            
        except Exception as e:
            print(f"  ❌ PubChem lookup failed: {e}")
            return None
    
    def get_chembl_data(self, compound_name):
        """Get bioactivity data from ChEMBL database"""
        try:
            # Search ChEMBL for compound
            search_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_synonyms__molecule_synonym__iexact={compound_name}"
            
            response = requests.get(search_url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data['molecules']:
                    mol_id = data['molecules'][0]['molecule_chembl_id']
                    
                    # Get bioactivity data
                    activity_url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id={mol_id}&target_type=PROTEIN&limit=10"
                    
                    activity_response = requests.get(activity_url, timeout=15)
                    if activity_response.status_code == 200:
                        activity_data = activity_response.json()
                        
                        bioactivities = []
                        for activity in activity_data['activities'][:5]:  # Limit to first 5
                            bioactivities.append({
                                'standard_type': activity.get('standard_type'),
                                'standard_value': activity.get('standard_value'),
                                'standard_units': activity.get('standard_units'),
                                'target_pref_name': activity.get('target_pref_name')
                            })
                        
                        chembl_data = {
                            'chembl_id': mol_id,
                            'bioactivities': bioactivities,
                            'source': 'ChEMBL Database'
                        }
                        
                        print(f"  ✅ ChEMBL data retrieved")
                        return chembl_data
                        
            print(f"  ⚠️ ChEMBL: No data found for {compound_name}")
            return None
            
        except Exception as e:
            print(f"  ❌ ChEMBL lookup failed: {e}")
            return None
    
    def validate_with_swissadme(self, smiles):
        """
        Note: SwissADME doesn't have a public API, but you can manually validate at:
        http://www.swissadme.ch/
        
        This function provides instructions for manual validation.
        """
        print(f"  📝 For SwissADME validation:")
        print(f"     1. Visit: http://www.swissadme.ch/")
        print(f"     2. Enter SMILES: {smiles}")
        print(f"     3. Compare results with your predictions")
        
        return {
            'manual_validation_url': 'http://www.swissadme.ch/',
            'smiles_to_test': smiles,
            'source': 'SwissADME (Manual Validation Required)'
        }
    
    def compare_predictions(self, compound_name):
        """Compare your predictions with external validation data"""
        if compound_name not in self.validation_results:
            print(f"No validation data found for {compound_name}")
            return
        
        result = self.validation_results[compound_name]
        your_preds = result['your_predictions']
        external = result['external_validation']
        
        print(f"\n📊 Validation Comparison for {compound_name}")
        print("=" * 60)
        
        # Compare molecular descriptors
        for source, data in external.items():
            if source in ['RDKit', 'PubChem']:
                print(f"\n{source} Comparison:")
                
                # Molecular Weight
                if 'molecular_weight' in data and 'molecular_weight' in your_preds:
                    ext_mw = data['molecular_weight']
                    your_mw = your_preds['molecular_weight']
                    diff = abs(ext_mw - your_mw) if ext_mw and your_mw else 'N/A'
                    print(f"  Molecular Weight: Your={your_mw}, {source}={ext_mw}, Diff={diff}")
                
                # LogP
                if 'logp' in data and 'logp' in your_preds:
                    ext_logp = data['logp']
                    your_logp = your_preds['logp']
                    diff = abs(ext_logp - your_logp) if ext_logp and your_logp else 'N/A'
                    print(f"  LogP: Your={your_logp}, {source}={ext_logp}, Diff={diff}")
                
                # TPSA
                if 'tpsa' in data and 'tpsa' in your_preds:
                    ext_tpsa = data['tpsa']
                    your_tpsa = your_preds['tpsa']
                    diff = abs(ext_tpsa - your_tpsa) if ext_tpsa and your_tpsa else 'N/A'
                    print(f"  TPSA: Your={your_tpsa}, {source}={ext_tpsa}, Diff={diff}")
        
        # Show bioactivity data from ChEMBL
        if 'ChEMBL' in external:
            chembl_data = external['ChEMBL']
            print(f"\nChEMBL Bioactivity Data:")
            for activity in chembl_data.get('bioactivities', []):
                print(f"  {activity['standard_type']}: {activity['standard_value']} {activity['standard_units']} vs {activity['target_pref_name']}")
    
    def generate_validation_report(self, output_file='validation_report.json'):
        """Generate a comprehensive validation report"""
        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"\n📄 Validation report saved to: {output_file}")
        return output_file

def main():
    """Example usage of the QSAR Validator"""
    
    # Initialize validator
    validator = QSARValidator()
    
    # Example compounds from your phytochemical database
    test_compounds = [
        {
            'name': 'Gallic acid',
            'smiles': 'C1=C(C=C(C(=C1O)O)O)C(=O)O',
            'your_predictions': {
                'molecular_weight': 170.12,
                'logp': 0.502,
                'tpsa': 97.99,
                'bioactivity_score': 4.826,
                'hbd': 4,
                'hba': 5
            }
        },
        {
            'name': 'Quercetin',
            'smiles': 'C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O',
            'your_predictions': {
                'molecular_weight': 302.238,
                'logp': 1.988,
                'tpsa': 131.36,
                'bioactivity_score': 4.796,
                'hbd': 5,
                'hba': 7
            }
        }
    ]
    
    print("🔬 QSAR Validation Tool")
    print("======================")
    print("This tool validates your QSAR predictions against external databases")
    
    # Validate each compound
    for compound in test_compounds:
        validator.validate_compound_properties(
            compound['name'],
            compound['smiles'],
            compound['your_predictions']
        )
        
        time.sleep(2)  # Be respectful to APIs
    
    # Compare predictions
    for compound in test_compounds:
        validator.compare_predictions(compound['name'])
    
    # Generate report
    validator.generate_validation_report()
    
    print("\n✅ Validation Complete!")
    print("\nRecommendations:")
    print("1. Check differences > 10% for molecular descriptors")
    print("2. Use ChEMBL bioactivity data to validate your bioactivity scores")
    print("3. Manually validate complex compounds using SwissADME")
    print("4. Consider literature search for experimental validation")

if __name__ == "__main__":
    main()