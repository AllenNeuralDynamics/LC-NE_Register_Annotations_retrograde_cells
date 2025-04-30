import os
import json
import pickle
from utils import utils

import numpy as np

def orient_mesh(
    vert_list,
    reg_dims: list,
    ds: int,
    orient: str,
    orient_matrix: np.ndarray,
    institute: str,
):
    """
    Imports cell locations from a XML file of cell likelihoods.

    Parameters
    -------------
    cell_likelihoods_path: str or PathLike
        Path to the cell likelihoods XML file.
    reg_dims: list
        Resolution (pixels) of the image used for segmentation, ordered relative to zarr.
    ds: int
        Factor by which the image for registration was downsampled from input_dims.
    orient: str
        The orientation the brain was imaged.
    orient_matrix: np.ndarray
        The direction of the axis of input cells relative to registration
    institute: str
        The institution that imaged the dataset.

    Returns
    -------------
    np.ndarray
        Array with cell locations scaled and oriented
    """
    cells = []

    for row in vert_list:
        
        x, y, z = int(row[2]), int(row[1]), int(row[0])
    
        # Corrects for a bug in acquisition as SPL is not an actual imaging orientation
        if orient == "sal":
            y = reg_dims[1] - (y / ds)
        else:
            y = y / ds
    
        cells.append(
                (z / ds, y, x / ds)
            )
    
    cells = np.array(cells)
    
    for idx, dim_orient in enumerate(orient_matrix.sum(axis = 1)):
        if dim_orient < 0:
            cells[:, idx] = reg_dims[idx] - cells[:, idx]

    return cells


def get_orientation(params: dict):
    """
    Fetch aquisition orientation to identify origin for cell locations
    from cellfinder. Important for read_xml function in quantification
    script

    Parameters
    ----------
    params : dict
        The orientation information from processing_manifest.json

    Returns
    -------
    orient : str
        string that indicates axes order and direction current available
        options are:
            'spr'
            'sal'
        But more may be used later
    """

    orient = ["", "", ""]
    for vals in params:
        direction = vals["direction"].lower()
        dim = vals["dimension"]
        orient[dim] = direction[0]

    return "".join(orient)

def get_orientation_transform(orientation_in: str, orientation_out: str):
    """
    Takes orientation acronyms (i.e. spr) and creates a convertion matrix for
    converting from one to another

    Parameters
    ----------
    orientation_in : str
        the current orientation of image or cells (i.e. spr)
    orientation_out : str
        the orientation that you want to convert the image or
        cells to (i.e. ras)

    Returns
    -------
    tuple
        the location of the values in the identity matrix with values
        (original, swapped)
    """

    reverse_dict = {"r": "l", "l": "r", "a": "p", "p": "a", "s": "i", "i": "s"}

    input_dict = {dim.lower(): c for c, dim in enumerate(orientation_in)}
    output_dict = {dim.lower(): c for c, dim in enumerate(orientation_out)}

    transform_matrix = np.zeros((3, 3))
    for k, v in input_dict.items():
        if k in output_dict.keys():
            transform_matrix[v, output_dict[k]] = 1
        else:
            k_reverse = reverse_dict[k]
            transform_matrix[v, output_dict[k_reverse]] = -1

    if orientation_in.lower() == "spl" or orientation_out.lower() == "spl":
        transform_matrix = abs(transform_matrix)

    original, swapped = np.where(transform_matrix.T)

    return original, swapped, transform_matrix

def get_region_lists():
    """
    Import list of acronyms of brain regions
    """
    
    CCF_dir = '../code/ccf_files/CCF_meshes'
    
    
    # Reading non-crossing structures to get acronyms
    with open(os.path.join(CCF_dir, "non_crossing_structures"), "rb") as f:
        hemi_struct = pickle.load(f)
        hemi_struct.remove(1051)  # don't know why this is being done
        hemi_labeled = [(s, "hemi") for s in hemi_struct]

    # Reading mid-crossing structures to get acronyms
    with open(os.path.join(CCF_dir, "mid_crossing_structures"), "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        mid_struct = u.load()
        mid_labeled = [(s, "mid") for s in mid_struct]

    return hemi_labeled + mid_labeled

def load_json_mesh(root, struct):
    """
    Loads meshes that are stored in json files like the CCF meshes

    Parameters
    ----------
    root : str
        path to mesh

    Returns
    -------
    tuple
        The vertices and faces of a given mesh

    """
        
    region_metadata = get_region_lists()
    region_dict = {i[0]: i[1] for i in region_metadata}
    fname = os.path.join(root, f"{struct}.json")
        
    with open(fname) as f:
        structure_data = json.loads(f.read())
        verts, faces = (
            np.array(structure_data[struct]["vertices"]),
            np.array(structure_data[struct]["faces"]),
        )
            
        if region_dict[int(struct)] == 'hemi':
            offset = faces.max() + 1
            faces = np.vstack((faces, faces + offset))
                
            verts_2 = verts.copy()
            verts_2[:, 0] = verts_2[:, 0] + (5700 - verts_2[:, 0]) * 2
            verts = np.vstack((verts, verts_2))
            
    return verts, faces

def warp_mesh(verts, faces, ccf_template, smartspim_template, ccf_transforms, template_transforms):
        
    # Transform to template
    ccf_params = utils.get_template_info(ccf_template)
    ants_verts = utils.convert_to_ants_space(ccf_params, verts)
    template_verts = utils.apply_transforms_to_points(
        ants_verts, ccf_transforms, invert=(False, False)
    )

    # Transform to lightsheet
    template_params = utils.get_template_info(smartspim_template)
    ls_verts = utils.apply_transforms_to_points(
        template_verts,
        template_transforms,
        invert=(False, False),
    )
    converted_verts = utils.convert_from_ants_space(template_params, ls_verts)
        
    return converted_verts, faces

def add_annotation_layer(ng_dict, cells):
    
    ng_cells = []
    for c, cell in enumerate(cells):
        ng_cells.append(
            {
                "point": [float(cell[0]), float(cell[1]), float(cell[2]), 0.5],
                "type": "point",
                "id": f"cell_{int(c)}"
            }
        )
    
    cropped_layer = {
        "type": "annotation",
        "source": {
            "url": "local://annotations",
            "transform": {
                "outputDimensions": {
                    "z": [0.000002, "m"],
                    "y": [0.0000018, "m"],
                    "x": [0.0000018, "m"],
                    "t": [0.001, "s"],
                },
            },
            
        },
        "tool": "annotatePoint",
        "tab": "annotations",
        "annotationColor": "#ff0000",
        "annotations": ng_cells,
        "name": "Pons Cells"
    }
    
    ng_dict['layers'].append(cropped_layer)
    return ng_dict


def dilate_mesh(mesh, dilation):
    
    if not isinstance(dilation, list):
        dilation = [dilation] * 3
        
    com = mesh.center_of_mass()
    
    verts, faces = mesh.points(), mesh.faces()
    
    for c, v in enumerate(verts):
        for i in range(3):
            v[i] = (v[i] - com[i]) * dilation[i] + com[i]
        verts[c, :] = v
    
    return vedo.Mesh([verts, faces])

class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(data, file_name):
    
    with open(file_name, 'w') as fp:
        json.dump(data, fp, indent = 2, cls=NumpyTypeEncoder)

def save_coordinates_with_indices_to_csv(coordinates, lt_id, output_dir='../results/'):
    """
    Save the transformed coordinates and row indices of cells to a CSV file.
    
    Parameters:
    - coordinates: The array of transformed coordinates to be saved.
    - lt_id: The identifier for the dataset (used as part of the filename).
    - output_dir: The directory where the CSV file will be stored.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create an array of row indices (0, 1, 2, ...)
    indices = np.arange(len(coordinates))
    
    # Combine the indices and coordinates (indices as the first column, then X, Y, Z)
    coordinates_with_indices = np.column_stack((indices, coordinates))
    
    # Construct the file path
    file_path = os.path.join(output_dir, f"{lt_id}_ccf.csv")
    
    # Save the coordinates with indices as a CSV file
    np.savetxt(file_path, coordinates_with_indices, delimiter=',', header="Index,X,Y,Z", comments='')
    
    print(f"Saved coordinates and indices for {lt_id} to {file_path}\n\n")

def rgb_to_hex(r,g,b):
    # Convert to a hexadecimal string
    hex_color = f'{r:02x}{g:02x}{b:02x}'
    # Convert the hexadecimal string to an integer in base-16
    color_int = int(hex_color, 16)
    return color_int