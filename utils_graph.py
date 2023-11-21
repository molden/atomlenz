import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from Object_Smiles import Objects_Smiles 
import copy
from rdkit import Chem
import pytorch_lightning as pl

def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
       for at_idx in at_matches_list:
           atom = mol.GetAtomWithIdx(at_idx)
           chg = atom.GetFormalCharge()
           hcount = atom.GetTotalNumHs()
           atom.SetFormalCharge(0)
           atom.SetNumExplicitHs(hcount - chg)
           atom.UpdatePropertyCache()
    return mol

def bb_box_intersects(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    #import ipdb; ipdb.set_trace()
    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    else:
        return 1

def dist_filter_bboxes(list_bboxes):
    distance_val = 0
    for index, bbox in enumerate(list_bboxes):
        for i in range(len(list_bboxes)-index-1):
            center1x = (bbox[0]+bbox[2])/2
            center1y = (bbox[1]+bbox[3])/2
            center2x = (list_bboxes[index+i+1][0]+list_bboxes[index+i+1][2])/2
            center2y = (list_bboxes[index+i+1][1]+list_bboxes[index+i+1][3])/2
            new_distance = distance([center1x,center1y],[center2x,center2y])
            if new_distance>distance_val :
               distance_val=new_distance
               cand1 = index
               cand2 = i+1
    return cand1, cand2

def tensordelete(tensor, indices):
    all_index = range(tensor.shape[0])
    res = [i for i in all_index if i not in indices]
    #import ipdb; ipdb.set_trace()
    if len(tensor.shape) == 2:
       return tensor[res,:]
    else:
       return tensor[res]
    #mask = torch.ones(tensor.shape, dtype=torch.bool)
    #if len(indices) > 0:
    #   index = torch.tensor(indices)
    #   mask.index_fill_(0,index,False)
    #    #print(tensor.shape)
    #    #print(mask.shape)
    #return tensor[mask]

def iou_filter_bboxes(list_bboxes, list_labels, list_scores, iou=0.5):
    #retrurn indexes of boxes with iou value > iou
    temp_indexes = {}
    filtered_list_boxes = list_bboxes
    filtered_list_labels = list_labels
    for index, bbox in enumerate(list_bboxes):
        for i in range(len(list_bboxes)-index-1):
            calc_iou = bb_intersection_over_union(bbox,list_bboxes[index+i+1])
            if calc_iou > iou:
               if f"{index}" in temp_indexes:
                  temp_indexes[f"{index}"].append(index+i+1)
               else:
                  temp_indexes[f"{index}"] = []
                  temp_indexes[f"{index}"].append(index+i+1)
    toremove_indexes = []
    for key in temp_indexes:
        index=int(key)
        score_index = list_scores[index]
        for next_index in temp_indexes[key]:
          #  import ipdb; ipdb.set_trace()
            if list_scores[next_index] > score_index:
                toremove_indexes.append(index)
                index = next_index
                score_index = list_scores[next_index]
            else:
                toremove_indexes.append(next_index)
    filtered_list_boxes = tensordelete(filtered_list_boxes,toremove_indexes)
    filtered_list_labels = tensordelete(filtered_list_labels,toremove_indexes)
    return filtered_list_boxes, filtered_list_labels

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou   

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def generateSmiles(molecule):
    dict_atom = {0: 'empty', 1: 'C', 2: 'H', 3: 'N', 4: 'O', 5: 'S', 6: 'F', 7: 'Cl', 8: 'Br', 9: 'I', 10: 'Se', 11: 'P', 12: 'B', 13: 'Si', 14:'Si', 15:'Si', 16: 'Si', 17:'Si'}
    #dict_atom = {0: 'empty', 2: 'C', 3: 'H', 4: 'N', 5: 'O', 6: 'S', 7: 'F', 8: 'Cl', 9: 'Br', 10: 'I', 11: 'Se', 12: 'P', 13: 'B', 14: 'Si', 18:'Si', 15:'Si', 16: 'Si', 17:'Si'}
    #dict_bond = {'1.0': 15, '2.0': 16, '3.0': 17, '4.0': 15, '4.5': 15, '5.0': 15, '5.5': 15}
    
    mol = Chem.RWMol()
    m = Chem.Mol()
    em = Chem.EditableMol(m)

    atom_labels = molecule['atom_labels']
    atom_list = []
    #create all atoms in RDKIT
    for index, atom_label in enumerate(atom_labels):
        #a = Chem.Atom(dict_atom[int(atom_label)-1])
        a = Chem.Atom(dict_atom[int(atom_label)])
        idx = em.AddAtom(a)
        atom_list.append(idx)
    #create all bond in RDKIT
    for index1, atom_label in enumerate(atom_labels):
        for index2,element in enumerate(molecule['graph'][index1][index1+1:]):
            if element > 1:

               #if element == 16:
               if element == 2 or element == 5 or element == 6:
                  bondIdx = em.AddBond(atom_list[index1],atom_list[index2+index1+1], Chem.BondType.SINGLE)
               #if element == 17:
               if element == 3:
                  bondIdx = em.AddBond(atom_list[index1],atom_list[index2+index1+1], Chem.BondType.DOUBLE) 
               #if element == 18:
               if element == 4:
         #         import ipdb; ipdb.set_trace()
                  bondIdx = em.AddBond(atom_list[index1],atom_list[index2+index1+1], Chem.BondType.TRIPLE)
    return Chem.MolToSmiles(em.GetMol())

def create_atom_labels(molecule, file_path):
    atom_labels=molecule['atom_labels']
    atom_boxes =molecule['atom_boxes']
    with open(file_path, "w") as fp:
       fp.write("label,xmin,ymin,xmax,ymax\n")
       for atom_label, atom_box in zip(atom_labels,atom_boxes):
           xmin,ymin,xmax,ymax = atom_box
           to_write=f"{atom_label-1},{xmin},{ymin},{xmax},{ymax}\n"
           fp.write(to_write)

def create_bond_labels(molecule, file_path):
    atom_boxes =molecule['atom_boxes']
    atom_labels=molecule['atom_labels']
    bond_list=[]
    for index1, atom_label in enumerate(atom_labels):
        for index2,element in enumerate(molecule['graph'][index1][index1+1:]):
            if element > 1:
               bond_list.append([index1,index2+index1+1,element])
    with open(file_path, "w") as fp:
       fp.write("label,xmin,ymin,xmax,ymax\n")
       for bond in bond_list:
           atom_box1 = atom_boxes[bond[0]]
           atom_box2 = atom_boxes[bond[1]]
           xmin1,ymin1,xmax1,ymax1= atom_box1
           xmin2,ymin2,xmax2,ymax2= atom_box2
           xmin = min(xmin1,xmin2)
           ymin = min(ymin1,ymin2)
           xmax = max(xmax1,xmax2)
           ymax = max(ymax1,ymax2)
           bond_type = bond[2]   
           to_write=f"{bond_type-1},{xmin},{ymin},{xmax},{ymax}\n"
           fp.write(to_write) 

def create_atom_bond_labels(molecule, file_path, image_id, transform_pars):
    dict_atom = {0: 'empty', 1: 'C', 2: 'H', 3: 'N', 4: 'O', 5: 'S', 6: 'F', 7: 'Cl', 8: 'Br', 9: 'I', 10: 'Se', 11: 'P', 12: 'B', 13: 'Si', 14:'*', 15:'Te', 16: 'Sn', 17:'As', 18:'Al', 19:'Ge', 20:'H', 21:'H'}
    #dict_atom = {0: 'empty', 1: 'C', 2: 'H', 3: 'N', 4: 'O', 5: 'S', 6: 'F', 7: 'Cl', 8: 'Br', 9: 'I', 10: 'Se', 11: 'P', 12: 'B', 13: 'Si', 14:'Si', 15:'Si', 16: 'Si', 17:'Si'}
    dict_charges = {1: '0', 2: '+1', 3: '-1', 4: '+2', 5: '-2', 6: '+3', 7: '+4', 8:'+5', 9: '+6'}
    dict_bond = {2: '1', 3:'2', 4: '3', 5: '1', 6: '4', 7: '5'} 
    atom_boxes =molecule['atom_boxes']
    atom_labels=molecule['atom_labels']
    atom_charges=molecule['charge_atoms']
    stereo_atoms = molecule['stereo_atoms']
    bond_list=[]
    linenumber=0
    x_scale,y_scale,xpad,ypad=transform_pars
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        fp = open(file_path,"a")
    else:
        fp = open(file_path,"w")
        fp.write("index,molid,bondtype,atom1,charge1,atom2,charge2,atom1coord1,atom1coord2,atom2coord1,atom2coord2\n")
    for index1, atom_label in enumerate(atom_labels):
        if atom_label>1:
           atom_box = atom_boxes[index1]
           xmin,ymin,xmax,ymax= atom_box
           xmin=int(np.round(xmin*x_scale))+xpad
           ymin=int(np.round(ymin*y_scale))+ypad
           xmax=int(np.round(xmax*x_scale))+xpad
           ymax=int(np.round(ymax*y_scale))+ypad
           atom_x = (xmin+xmax)/2
           atom_y = (ymin+ymax)/2
           atom_charge = atom_charges[index1]
           fp.write(f"{image_id}{linenumber},{image_id},nobond,{dict_atom[int(atom_label)]},{dict_charges[int(atom_charge)]},{dict_atom[int(atom_label)]},{dict_charges[int(atom_charge)]},{atom_x},{atom_y},{atom_x},{atom_y}\n")
           linenumber+=1
        for index2,element in enumerate(molecule['graph'][index1][index1+1:]):
            if element > 1:
               bond_list.append([index1,index2+index1+1,element])
    for j,bond in enumerate(bond_list):
        atom1_label=atom_labels[bond[0]]
        atom2_label=atom_labels[bond[1]]
        atom_box1 = atom_boxes[bond[0]]
        atom_box2 = atom_boxes[bond[1]]
        xmin1,ymin1,xmax1,ymax1= atom_box1
        xmin1=int(np.round(xmin1*x_scale))+xpad
        ymin1=int(np.round(ymin1*y_scale))+ypad
        xmax1=int(np.round(xmax1*x_scale))+xpad
        ymax1=int(np.round(ymax1*y_scale))+ypad
        xmin2,ymin2,xmax2,ymax2= atom_box2
        xmin2=int(np.round(xmin2*x_scale))+xpad
        ymin2=int(np.round(ymin2*y_scale))+ypad
        xmax2=int(np.round(xmax2*x_scale))+xpad
        ymax2=int(np.round(ymax2*y_scale))+ypad
        atom1_x=(xmin1+xmax1)/2
        atom1_y=(ymin1+ymax1)/2
        atom2_x=(xmin2+xmax2)/2
        atom2_y=(ymin2+ymax2)/2
        atom1_charge=atom_charges[bond[0]]
        atom2_charge=atom_charges[bond[1]]
        suffix='.0'
        if bond[2] > 5:
            if stereo_atoms[bond[0]]==1:
                suffix='.0'
            else:
                suffix='.5'
        fp.write(f"{image_id}{linenumber+j+1},{image_id},{dict_bond[int(bond[2])]}{suffix},{dict_atom[int(atom1_label)]},{dict_charges[int(atom1_charge)]},{dict_atom[int(atom2_label)]},{dict_charges[int(atom2_charge)]},{atom1_x},{atom1_y},{atom2_x},{atom2_y}\n")

def create_stereo_labels(molecule, file_path):
    atom_boxes =molecule['atom_boxes']
    stereo_atoms = molecule['stereo_atoms']
    with open(file_path, "w") as fp:
       fp.write("label,xmin,ymin,xmax,ymax\n")
       for i,stereo in enumerate(stereo_atoms):
           #if stereo == 1:
              xmin,ymin,xmax,ymax= atom_boxes[i]
              to_write=f"{stereo},{xmin},{ymin},{xmax},{ymax}\n"
              fp.write(to_write)

def create_charge_labels(molecule, file_path):
    atom_boxes = molecule['atom_boxes']
    atom_charges=molecule['charge_atoms']
    with open(file_path, "w") as fp:
       fp.write("label,xmin,ymin,xmax,ymax\n")
       for i,charge in enumerate(atom_charges):
           xmin,ymin,xmax,ymax= atom_boxes[i]
           to_write=f"{charge-1},{xmin},{ymin},{xmax},{ymax}\n"
           fp.write(to_write)

def save_mol_to_file(molecule, file_path, return_mol_info=0,height=400,width=400):
    #{6: 1, 1: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7, 35: 8, 53: 9, 34: 10, 15: 11, 5: 12, 14: 13, 52: 15, 50: 16, 33: 17, 13: 18, 32: 19, 'H2': 20, 'H3':21}
    #dict_atom = {0: 'empty', 1: 'C', 2: 'H', 3: 'N', 4: 'O', 5: 'S', 6: 'F', 7: 'Cl', 8: 'Br', 9: 'I', 10: 'Se', 11: 'P', 12: 'B', 13: 'Si', 14:'Si', 15:'Si', 16: 'Si', 17:'Si'}
    dict_atom = {0: 'empty', 1: 'C', 2: 'H', 3: 'N', 4: 'O', 5: 'S', 6: 'F', 7: 'Cl', 8: 'Br', 9: 'I', 10: 'Se', 11: 'P', 12: 'B', 13: 'Si', 14:'*', 15:'Te', 16: 'Sn', 17:'As', 18:'Al', 19:'Ge', 20:'H', 21:'H'}
    #dict_charges = {'0': 0, '1': 1, '-1': 2, '2': 3, '-2': 4, '3': 5, '4': 6, '5': 7, '6': 8}
    dict_charges = {1: 0, 2: 1, 3: -1, 4: 2, 5: -2, 6: 3, 7: 4, 8: 5, 9: 6}
    file = open(file_path,"w")
    file.write("\n")
    file.write(" MOLFILE FROM CHEMGRAPHER\n")
    file.write("\n")
    corrected_molecule = copy.deepcopy(molecule)
    atom_labels=molecule['atom_labels']
    atom_boxes =molecule['atom_boxes']
    atom_charges=molecule['charge_atoms']
    bond_list=[]
    mol_atom_list=[]
    mol_bond_list=[]
    atom_check_bonds=np.zeros(len(atom_labels))
    for index1, atom_label in enumerate(atom_labels):
        for index2,element in enumerate(molecule['graph'][index1][index1+1:]):
            if element > 1:
               bond_list.append([index1,index2+index1+1,element])
               atom_check_bonds[index1]=1
               atom_check_bonds[index2+index1+1]=1
    num_atoms = len(atom_labels)
    num_bonds = len(bond_list)
    stereo_atoms = corrected_molecule['stereo_atoms']
    if np.any(stereo_atoms):
        chiral=1
    else:
        chiral=0
    #file.write(f"{num_atoms:3.0f}{num_bonds:3.0f}  0  0  {chiral}  0            999 V2000\n")
    atom_block=""
    bond_block=""
    atom_translate=np.zeros(len(atom_labels))
    translate_index=0
    for i,atom in enumerate(atom_labels):
        atom_box = atom_boxes[i]
        xmin, ymin, xmax, ymax =atom_box
        #centerx = ((xmin+xmax)/2)-200
        centerx = ((xmin+xmax)/2-width/2)/width/2
        #should be ((xmin+xmax)/2-xwidth/2)/xwidth/2
        #centery = ((ymin+ymax)/2-200)*(-1)
        centery = ((ymin+ymax)/2-height/2)*(-1)/(height/2)
        #should be ((ymin+ymax)/2-yheight/2)*(-1)/yheight/2
        zero = 0.0
        #print(f"centerx = {centerx}, centery = {centery}, zero={zero}")
        if atom_check_bonds[i]==1:
           atom_translate[i]=translate_index
           atom_block+=f"{centerx:10.4f}{centery:10.4f}{zero:10.4f}  {dict_atom[int(atom)]}  0  0\n"
           #file.write(f"{centerx:10.4f}{centery:10.4f}{zero:10.4f}  {dict_atom[int(atom)]}  0  0\n")
           mol_atom_list.append([float(centerx),float(centery)])
           translate_index+=1
    for j,bond in enumerate(bond_list):
        bond_label = bond[2]
        if bond_label > 5:
            if stereo_atoms[bond[0]]==1:
                atom1 = bond[0]
                atom2 = bond[1]
            else:
                stereo_atoms[bond[1]]=1
                atom1 = bond[1]
                atom2 = bond[0]
            bond_num = 1
        else:
            atom1 = bond[0]
            atom2 = bond[1]
            bond_num = bond_label - 1

        if bond_label == 6:
           bond_dir = 1
        elif bond_label == 7:
           bond_dir = 6
        else:
           bond_dir = 0
        atom1_translated=atom_translate[atom1]
        atom2_translated=atom_translate[atom2]
        #bond_block+=f"{atom1+1:3.0f}{atom2+1:3.0f}{bond_num:3.0f}{bond_dir:3.0f}\n"
        bond_block+=f"{atom1_translated+1:3.0f}{atom2_translated+1:3.0f}{bond_num:3.0f}{bond_dir:3.0f}\n"
        #file.write(f"{atom1+1:3.0f}{atom2+1:3.0f}{bond_num:3.0f}{bond_dir:3.0f}\n")
        mol_bond_list.append([atom1+1,atom2+1,int(bond_num)])
    corrected_molecule['stereo_atoms'] = stereo_atoms
    header=f"{len(mol_atom_list):3.0f}{len(mol_bond_list):3.0f}  0  0  {chiral}  0            999 V2000\n"
    file.write(header)
    file.write(atom_block)
    file.write(bond_block)
    for i,atom in enumerate(atom_labels):
        if atom != 1 and atom_check_bonds[i]==1:
            file.write(f"M  CHG  1{i+1:3.0f}{dict_charges[int(atom_charges[i])]:3.0f}\n")
    for i,atom in enumerate(atom_labels):
        if atom == 20 and atom_check_bonds[i]==1:
            file.write(f"M  ISO  1{i+1:3.0f}{2:3.0f}\n")
            if file_path == "molfile_new":
               import ipdb; ipdb.set_trace()
        if atom == 21 and atom_check_bonds[i]==1:
            file.write(f"M  ISO  1{i+1:3.0f}{3:3.0f}\n")
            if file_path == "molfile_new":
               import ipdb; ipdb.set_trace()


    file.write(f"M  END")
    if return_mol_info>0:
        return mol_atom_list,mol_bond_list
    else:
        return corrected_molecule

           



def visualize_atom_bonds(impath, atoms_bboxes, bond_bboxes):
    im = plt.imread(str(impath))
    plt.imshow(im, cmap="gray")
    for bbox in atoms_bboxes:
        xmin, ymin, xmax, ymax =bbox
        plt.plot([xmin, xmin, xmax, xmax, xmin],
                 [ymin, ymax, ymax, ymin, ymin],
                 color="blue",
                 label="a")
    plt.savefig("example_image_atom.png")
    plt.clf()
    plt.imshow(im, cmap="gray")
    for bbox in bond_bboxes:
        xmin, ymin, xmax, ymax =bbox
        plt.plot([xmin, xmin, xmax, xmax, xmin],
                 [ymin, ymax, ymax, ymin, ymin],
                 color="blue",
                 label="a")
    plt.savefig("example_image_bond.png")
    plt.clf()

def alter_stereos(molecule, original_smiles, filesuffix=""):
    num_atoms = len(molecule['atom_labels'])
    original_mol = Chem.MolFromSmiles(original_smiles)
    if original_mol is not None:
       smiles_original = Chem.MolToSmiles(original_mol)
    else:
       smiles_original = original_smiles
    new_molecule = copy.deepcopy(molecule)
    for i in range(num_atoms):
        alter_stereo = copy.deepcopy(molecule['stereo_atoms'])
        alter_stereo[i] = 1-alter_stereo[i]
        file_path2 = f"molfiletest{filesuffix}"
        new_molecule['stereo_atoms']=alter_stereo
        save_mol_to_file(new_molecule, file_path2)
        predicted_mol = Chem.MolFromMolFile(f"molfiletest{filesuffix}")
        if predicted_mol is not None:
           try:
             smiles_pred_mol = Chem.MolToSmiles(predicted_mol)
           except RuntimeError:
             smiles_pred_mol = "predicted_mol"
        else:
           smiles_pred_mol = "predicted_mol"
        smiles_noslashes = smiles_pred_mol
        smiles_noslashes = smiles_noslashes.replace("\\","")
        smiles_noslashes = smiles_noslashes.replace("/","")
        if smiles_pred_mol == smiles_original or smiles_noslashes == smiles_original:
           return new_molecule
    return None

def alter_charges(molecule, original_smiles, filesuffix=""):
    num_atoms = len(molecule['atom_labels'])
    original_mol = Chem.MolFromSmiles(original_smiles)
    if original_mol is not None:
       smiles_original = Chem.MolToSmiles(original_mol)
    else:
       smiles_original = original_smiles
    new_molecule = copy.deepcopy(molecule)
    for i in range(num_atoms):
        alter_charges = copy.deepcopy(molecule['charge_atoms'])
        for type_key in range(9):
            alter_charges[i] = type_key+1
            file_path2 = f"molfiletest{filesuffix}"
            new_molecule['charge_atoms']=alter_charges
            save_mol_to_file(new_molecule, file_path2)
            predicted_mol = Chem.MolFromMolFile(f"molfiletest{filesuffix}")
            if predicted_mol is not None:
               try:
                 smiles_pred_mol = Chem.MolToSmiles(predicted_mol)
               except RuntimeError:
                 smiles_pred_mol = "predicted_mol"
            else:
               smiles_pred_mol = "predicted_mol"
            smiles_noslashes = smiles_pred_mol
            smiles_noslashes = smiles_noslashes.replace("\\","")
            smiles_noslashes = smiles_noslashes.replace("/","")
            if smiles_pred_mol == smiles_original or smiles_noslashes == smiles_original:
               return new_molecule
    return None


def alter_atoms(molecule, original_smiles, filesuffix="", replacenitrogen=0):
    num_atoms = len(molecule['atom_labels'])
    original_mol = Chem.MolFromSmiles(original_smiles)
    if original_mol is not None:
       smiles_original = Chem.MolToSmiles(original_mol)
    else:
       smiles_original = original_smiles
    new_molecule = copy.deepcopy(molecule)
    for i in range(num_atoms):
        alter_atoms = copy.deepcopy(molecule['atom_labels'])
        for type_key in range(21):
                alter_atoms[i] = type_key+1
                file_path2 = f"molfiletest{filesuffix}"
                new_molecule['atom_labels']=alter_atoms
                save_mol_to_file(new_molecule, file_path2)
                predicted_mol = Chem.MolFromMolFile(f"molfiletest{filesuffix}")
                if predicted_mol is not None:
                   if replacenitrogen == 1:
                     nitrogen = Chem.MolFromSmiles("[N+](=O)[O-]")
                     replaced = Chem.rdmolops.ReplaceSubstructs(predicted_mol,Chem.MolFromSmiles("*"),nitrogen,replaceAll=True)
                     predicted_mol = replaced[0]
                   #  smiles_pred_mol = Chem.MolToSmiles(predicted_mol)
                   try:
                     smiles_pred_mol = Chem.MolToSmiles(predicted_mol)
                   except RuntimeError:
                     smiles_pred_mol = "predicted_mol"
                else:
                   smiles_pred_mol = "predicted_mol"
                smiles_noslashes = smiles_pred_mol
                smiles_noslashes = smiles_noslashes.replace("\\","")
                smiles_noslashes = smiles_noslashes.replace("/","")
                if smiles_pred_mol == smiles_original or smiles_noslashes == smiles_original:
                   return new_molecule
    return None

def alter_bonds(molecule, original_smiles, filesuffix=""):
    original_mol = Chem.MolFromSmiles(original_smiles)
    if original_mol is not None:
       smiles_original = Chem.MolToSmiles(original_mol)
    else:
       smiles_original = original_smiles
    new_molecule = copy.deepcopy(molecule)
    atom_labels = molecule['atom_labels']
    bond_list=[]
    for index1, atom_label in enumerate(atom_labels):
        for index2,element in enumerate(molecule['graph'][index1][index1+1:]):
            if element > 1:
               bond_list.append([index1,index2+index1+1,element])
    for bond in bond_list:
        alter_graph = copy.deepcopy(molecule['graph'])
        for bond_type in [2,3,4,5,6,7]:
            alter_graph[bond[0],bond[1]]=bond_type
            alter_graph[bond[1],bond[0]]=bond_type
            new_molecule['graph']=alter_graph
            file_path2 = f"molfiletestbond{filesuffix}"
            save_mol_to_file(new_molecule, file_path2)
            predicted_mol = Chem.MolFromMolFile(f"molfiletestbond{filesuffix}")
            if predicted_mol is not None:
               try:
                 smiles_pred_mol = Chem.MolToSmiles(predicted_mol)
               except RuntimeError:
                 smiles_pred_mol = "predicted_mol"
            else:
               smiles_pred_mol = "predicted_mol"
            smiles_noslashes = smiles_pred_mol
            smiles_noslashes = smiles_noslashes.replace("\\","")
            smiles_noslashes = smiles_noslashes.replace("/","")
            if smiles_pred_mol == smiles_original or smiles_noslashes == smiles_original:
               return new_molecule
    return None

            
def make_predictions(model_cls,args,atom_model_path, bond_model_path, charge_model_path, stereo_model_path, dataset):
    experiment_path_atoms = atom_model_path
    dir_list = os.listdir(experiment_path_atoms)
    dir_list = [os.path.join(experiment_path_atoms,f) for f in dir_list]
    dir_list.sort(key=os.path.getctime, reverse=True)
    checkpoint_file_atoms = [f for f in dir_list if "ckpt" in f][0]
    dataset.prepare_data()
    model_atom = model_cls.load_from_checkpoint(checkpoint_file_atoms)
    model_atom.model.roi_heads.score_thresh = args.score_thresh
    trainer = pl.Trainer(logger=False, gpus=1)
    atom_preds = trainer.predict(model_atom, dataset.test_dataloader())
    experiment_path_bonds = bond_model_path
    dir_list = os.listdir(experiment_path_bonds)
    dir_list = [os.path.join(experiment_path_bonds,f) for f in dir_list]
    dir_list.sort(key=os.path.getctime, reverse=True)
    checkpoint_file_bonds = [f for f in dir_list if "ckpt" in f][0]
    model_bond = model_cls.load_from_checkpoint(checkpoint_file_bonds)
    bond_preds = trainer.predict(model_bond, dataset.test_dataloader())
    experiment_path_stereo = stereo_model_path
    dir_list = os.listdir(experiment_path_stereo)
    dir_list = [os.path.join(experiment_path_stereo,f) for f in dir_list]
    dir_list.sort(key=os.path.getctime, reverse=True)
    checkpoint_file_stereo = [f for f in dir_list if "ckpt" in f][0]
    model_stereo = model_cls.load_from_checkpoint(checkpoint_file_stereo)
    stereo_preds = trainer.predict(model_stereo, dataset.test_dataloader()) 
    experiment_path_charges = charge_model_path
    dir_list = os.listdir(experiment_path_charges)
    dir_list = [os.path.join(experiment_path_charges,f) for f in dir_list]
    dir_list.sort(key=os.path.getctime, reverse=True)
    checkpoint_file_charges = [f for f in dir_list if "ckpt" in f][0]
    model_charges = model_cls.load_from_checkpoint(checkpoint_file_charges)
    charges_preds = trainer.predict(model_charges, dataset.test_dataloader())
    return atom_preds, bond_preds, charges_preds, stereo_preds

def create_molecules(atom_preds, bond_preds, charges_preds, stereo_preds):
    mol_graphs = []
    for image_idx, bonds in enumerate(bond_preds):
        atom_boxes = atom_preds[image_idx]['boxes'][0]
        atom_labels = atom_preds[image_idx]['preds'][0]
        atom_scores = atom_preds[image_idx]['scores'][0]
        charge_boxes = charges_preds[image_idx]['boxes'][0]
        charge_labels = charges_preds[image_idx]['preds'][0]
        charge_mask=torch.where(charge_labels>1)
        filtered_ch_labels=charge_labels[charge_mask]
        filtered_ch_boxes=charge_boxes[charge_mask]
        #import ipdb; ipdb.set_trace()
        filtered_bboxes, filtered_labels = iou_filter_bboxes(atom_boxes, atom_labels, atom_scores)
        #for atom_label in filtered_labels:
        #    count_atoms_preds[atom_label] += 1
        #import ipdb; ipdb.set_trace()
        mol_graph = np.zeros((len(filtered_bboxes),len(filtered_bboxes)))
        stereo_atoms = np.zeros(len(filtered_bboxes))
        charge_atoms = np.ones(len(filtered_bboxes))
        for index,box_atom in enumerate(filtered_bboxes):
            for box_charge,label_charge in zip(filtered_ch_boxes,filtered_ch_labels):
                if bb_box_intersects(box_atom,box_charge) == 1:
                    charge_atoms[index]=label_charge

        for bond_idx, bond_box in enumerate(bonds['boxes'][0]):
            label_bond = bonds['preds'][0][bond_idx]
            if label_bond > 1:
               #import ipdb; ipdb.set_trace()
              result = []
              limit = 0
            #TODO: values of 50 and 5 should be made dependent of mean size of atom_boxes
              while result.count(1) < 2 and limit < 80:
                 result=[]
                 bigger_bond_box = [bond_box[0]-limit,bond_box[1]-limit,bond_box[2]+limit,bond_box[3]+limit]
                 for atom_box in filtered_bboxes:
                     result.append(bb_box_intersects(atom_box,bigger_bond_box))
                 limit+=5
              indices = [i for i, x in enumerate(result) if x == 1]
              if len(indices) == 2:
               #import ipdb; ipdb.set_trace()
                 mol_graph[indices[0],indices[1]]=label_bond
                 mol_graph[indices[1],indices[0]]=label_bond
              if len(indices) > 2:
                #we have more then two canidate atoms for one bond, we filter ...
                  cand_bboxes = filtered_bboxes[indices,:]
                  cand_indices = dist_filter_bboxes(cand_bboxes)
                  mol_graph[indices[cand_indices[0]],indices[cand_indices[1]]]=label_bond
                  mol_graph[indices[cand_indices[1]],indices[cand_indices[0]]]=label_bond
        stereo_bonds = np.where(mol_graph>4, True, False)
        if np.any(stereo_bonds):
           stereo_boxes = stereo_preds[image_idx]['boxes'][0]
           stereo_labels= stereo_preds[image_idx]['preds'][0]
           for stereo_box,stereo_label in zip(stereo_boxes,stereo_labels):
              if stereo_label > 1: #to be used if on stereo are labeled 1 and stereo 2
                 result=[]
                 for atom_box in filtered_bboxes:
                     result.append(bb_box_intersects(atom_box,stereo_box))
                 indices = [i for i, x in enumerate(result) if x == 1] 
                 if len(indices) == 1:
                     stereo_atoms[indices[0]]=1

        molecule = dict()
        molecule['graph'] = mol_graph
        #molecule['atom_labels'] = atom_preds[image_idx]['preds'][0]
        molecule['atom_labels'] = filtered_labels
        molecule['atom_boxes'] = filtered_bboxes
        molecule['stereo_atoms'] = stereo_atoms
        molecule['charge_atoms'] = charge_atoms
        mol_graphs.append(molecule)
    return mol_graphs

def solve_mol_problems(mol,problems):
    em = Chem.EditableMol(mol)  
    bonds = mol.GetBonds()
    for problem in problems:  
        bonds = mol.GetBonds()
        not_yet_solved = True
        if problem.GetType() == 'AtomValenceException':
           prob_atom_idx = problem.GetAtomIdx()
           for bond in bonds:
              if bond.GetBeginAtomIdx()==prob_atom_idx and not_yet_solved:
                 a1 = bond.GetBeginAtomIdx()
                 a2 = bond.GetEndAtomIdx()
                 em.RemoveBond(a1,a2)
                 not_yet_solved=False
              if bond.GetEndAtomIdx()==prob_atom_idx and not_yet_solved:
                 a1 = bond.GetBeginAtomIdx()
                 a2 = bond.GetEndAtomIdx()
                 em.RemoveBond(a1,a2)
                 not_yet_solved=False
           mol = em.GetMol()
    return mol
