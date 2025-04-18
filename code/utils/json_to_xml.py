import os
import re
import json
import pickle
import xmltodict

from glob import glob
from imlib.cells.cells import Cell
from imlib.IO.cells import save_cells, get_cells

def read_json_as_dict(filepath: str):
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary

def get_annotations(annot_path, tp_name):
    """
    Get the true positives and false positives for ground truth cells

    Parameters
    ------------------------
    annot_path: Pathlike
        The location of the annotated neuroglancer data

    tp_name: str
        name of the annotation layer that has the true positives

    Returns
    ------------------------
    annotations: dict
        disctionary with cell locations broken up by type
    """

    if 'json' in annot_path:
        pass
    else:
        annot_path = glob(os.path.join(annot_path, "*.json"))[0]

    annot_dict = read_json_as_dict(annot_path)
    annot_layers = annot_dict['layers']

    annotations = {}

    for layer in annot_layers:
        if layer['type'] == 'annotation' and layer['name'].lower() == tp_name:
            annot_list = []
            for annotation in layer['annotations']:
                annot_list.append(
                    [
                        int(annotation['point'][0]),
                        int(annotation['point'][1]),
                        int(annotation['point'][2])
                    ]
                )
            
                annotations['cells'] = annot_list

    return annotations

def convert_to_cell_obj(cells):

    cells_transformed = []
    c_type = 1

    for cond, cells in cells.items():
        
        if len(cells) == 1:
            cells *= 2
        
        for cell in cells:

            cell_obj = Cell(
                (
                    cell[2],
                    cell[1],
                    cell[0]
                ),
                c_type
            )

            cells_transformed.append(cell_obj)
    
    return cells_transformed

def json_to_xml(params):
    
    files  = glob(os.path.join(params['path'], '*.json'))
    
    for file in files:
        
        fname = os.path.basename(file)
        out_path = os.path.join(params['save_path'], fname[:-4] + 'xml')
        
        cells = get_annotations(file, params['tp_name'])
        cell_obj_list = convert_to_cell_obj(cells)
        
        print(f"Saving cells from {file} to {out_path}")
        
        save_cells(cell_obj_list, out_path)
        
    return

def dict_to_xml(cells, save_fname):

    cell_obj_list = convert_to_cell_obj(cells)
    
    out_path = f"../results/{save_fname}"
    save_cells(cell_obj_list, out_path)
    
    print(f"Saving cells to {out_path}")
    
    return