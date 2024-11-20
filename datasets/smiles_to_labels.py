import os
import numpy as np
from rdkit import Chem

def main():
    import argparse
    atomnumber_dict = {6: 1, 1: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7, 35: 8, 53: 9, 34: 10, 15: 11, 5: 12, 14: 13, 19: 14}
    parser = argparse.ArgumentParser(description="Parse the SMILES labels")

    parser.add_argument("--inputfolder", help="Input folder with SMILES labels", type=str, default="smiles/")
    parser.add_argument("--outputfolder", help="Outputfolder where labels will be saved.", type=str, default="labels/")
    args = parser.parse_args()
    num_smiles = len([name for name in os.listdir(args.inputfolder) if os.path.isfile(f"{args.inputfolder}/{name}")])
    for fnum in range(num_smiles):
        with open(f"{args.inputfolder}/{fnum}.txt") as smilesfile:
          count_atoms = np.zeros(15)
          smiles = smilesfile.read()
        m        = Chem.MolFromSmiles(smiles,sanitize=False)
        for atom in m.GetAtoms():
            count_atoms[atomnumber_dict.get(atom.GetAtomicNum(),14)] +=1
        labels = count_atoms[:]
        with open(f"{args.outputfolder}/{fnum}.txt", "w") as fp:
                fp.write("label,xmin,ymin,xmax,ymax\n")
                for i_atom, label_count in enumerate(labels):
                    for i in range(int(label_count)):
                        to_write=f"{i_atom-1},0,0,0,0\n"
                        fp.write(to_write)

if __name__ == "__main__":
    main()
