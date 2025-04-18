import os
import copy
import json
import ants
import pims
import vedo
import numpy as np
import pandas as pd
import xmltodict
import dask.array as da
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from skimage import measure
from sklearn.metrics import normalized_mutual_info_score
from imlib.cells.cells import Cell
from imlib.IO.cells import get_cells, save_cells

from pathlib import Path
from typing import Union, List

PathLike = Union[Path, str]

def read_json_as_dict(filepath: str) -> dict:
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

def read_cells_from_xml(
    cell_likelihoods_path: Union[str, "PathLike"],
    reg_dims: List[int],
    ds: int,
    orient: str,
    orient_matrix: np.ndarray,
    institute: str,
) -> List[tuple]:
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
    if not Path(cell_likelihoods_path).exists():
        raise FileNotFoundError(f"Path {cell_likelihoods_path} does not exist.")

    cells_list = get_points_from_xml(cell_likelihoods_path)

    cells = []

    for row in cells_list:
        
        x, y, z = int(row["x"]), int(row["y"]), int(row["z"])
    
        # Corrects for a bug in acquisition as SPL is not an actual imaging orientation
        if orient == "spl" and institute == "AIBS":
            y = reg_dims[1] - (y / ds)
        else:
            y = y / ds
    
        cells.append(
                (z / ds, y, x / ds)
            )
    
    cells = np.array(cells)
    print(f"Cell Max: {np.max(cells, axis = 0)}")
    
    for idx, dim_orient in enumerate(orient_matrix.sum(axis = 1)):
        if dim_orient < 0:
            cells[:, idx] = reg_dims[idx] - cells[:, idx]
    
    print(f"Cell Max: {np.max(cells, axis = 0)}")
                              
    return cells

def read_cells_from_csv(
    cell_likelihoods_path: Union[str, "PathLike"],
    reg_dims: List[int],
    ds: int,
    orient: str,
    orient_matrix: np.ndarray,
    institute: str,
) -> List[tuple]:
    """
    Imports cell locations from a CSV file of cell likelihoods.

    Parameters
    -------------
    cell_likelihoods_path: str or PathLike
        Path to the cell likelihoods CSV file.
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
    if not Path(cell_likelihoods_path).exists():
        raise FileNotFoundError(f"Path {cell_likelihoods_path} does not exist.")

    df = pd.read_csv(cell_likelihoods_path)

    cells = []

    for _, row in df.iterrows():
        x, y, z = row["X"], row["Y"], row["Z"]

        # Corrects for a bug in acquisition as SPL is not an actual imaging orientation
        if orient == "spl" and institute == "AIBS":
            y = reg_dims[1] - (y / ds)
        else:
            y = y / ds

        cells.append((z / ds, y, x / ds))

    cells = np.array(cells)

    for idx, dim_orient in enumerate(orient_matrix.sum(axis=1)):
        if dim_orient < 0:
            cells[:, idx] = reg_dims[idx] - cells[:, idx]

    return cells

def get_points_from_xml(path: PathLike, encoding: str = "utf-8") -> List[dict]:
    """
    Function to parse the points from the
    cell segmentation capsule.

    Parameters
    -----------------

    Path: PathLike
        Path where the XML is stored.

    encoding: str
        XML encoding. Default: "utf-8"

    Returns
    -----------------
    List[dict]
        List with the location of the points.
    """

    with open(path, "r", encoding=encoding) as xml_reader:
        xml_file = xml_reader.read()

    xml_dict = xmltodict.parse(xml_file)
    cell_data = xml_dict["CellCounter_Marker_File"]["Marker_Data"][
        "Marker_Type"
    ]["Marker"]

    new_cell_data = []
    for cell in cell_data:
        new_cell_data.append(
            {"x": cell["MarkerX"], "y": cell["MarkerY"], "z": cell["MarkerZ"],}
        )

    return new_cell_data

def check_orientation(img: np.array, params: dict, orientations: dict):
    """
    Checks aquisition orientation an makes sure it is aligned to the CCF. The
    CCF orientation is:
        - superior_to_inferior
        - left_to_right
        - anterior_to_posterior

    Parameters
    ----------
    img : np.array
        The raw image in its aquired orientatin
    params : dict
        The orientation information from processing_manifest.json
    orientations: dict
        The axis order of the CCF reference atals

    Returns
    -------
    img_out : np.array
        The raw image oriented to the CCF
    """

    orient_mat = np.zeros((3, 3))
    acronym = ["", "", ""]

    for k, vals in enumerate(params):
        direction = vals["direction"].lower()
        dim = vals["dimension"]
        if direction in orientations.keys():
            ref_axis = orientations[direction]
            orient_mat[dim, ref_axis] = 1
            acronym[dim] = direction[0]
        else:
            direction_flip = "_".join(direction.split("_")[::-1])
            ref_axis = orientations[direction_flip]
            orient_mat[dim, ref_axis] = -1
            acronym[dim] = direction[0]

    # check because there was a bug that allowed for invalid spl orientation
    # all vals should be postitive so just taking absolute value of matrix
    if "".join(acronym) == "spl":
        orient_mat = abs(orient_mat)

    original, swapped = np.where(orient_mat)
    img_out = np.moveaxis(img, original, swapped)

    out_mat = orient_mat[:, swapped]
    for c, row in enumerate(orient_mat.T):
        val = np.where(row)[0][0]
        if row[val] == -1:
            img_out = np.flip(img_out, c)
            out_mat[val, val] *= -1

    return img_out, orient_mat, out_mat

def get_orientation(params: dict) -> str:
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


def get_orientation_transform(orientation_in: str, orientation_out: str) -> tuple:
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
    
    
    reverse_dict = {
        'r': 'l',
        'l': 'r',
        'a': 'p',
        'p': 'a',
        's': 'i',
        'i': 's'
    }
    
    input_dict = {dim.lower(): c for c, dim in enumerate(orientation_in)}
    output_dict = {dim.lower(): c for c, dim in enumerate(orientation_out)}
    
    transform_matrix = np.zeros((3,3))
    
    for k, v in input_dict.items():
        if k in output_dict.keys():
            transform_matrix[v, output_dict[k]] = 1
        else:
            k_reverse = reverse_dict[k]
            transform_matrix[v, output_dict[k_reverse]] = -1
    
    if orientation_in.lower() == "spl" or orientation_out.lower() == "spl" :
        transform_matrix = abs(transform_matrix)

    original, swapped = np.where(transform_matrix.T)
    
    return original, swapped, transform_matrix

def orient_image(img, orient_mat):
    
    original, swapped = np.where(orient_mat)
    img_out = np.moveaxis(img, original, swapped)
    
    for c, row in enumerate(orient_mat.T):
        val = np.where(row)[0][0]
        if row[val] == -1:
            img_out = np.flip(img_out, c)
    
    return img_out

def get_template_orientations(reg_path: PathLike) -> tuple:
    
    warp_file = os.path.abspath(os.path.join(reg_path, "ls_to_template_SyN_1Warp.nii.gz"))
    warp_template = ants.image_read(warp_file)
    
    warp_file = os.path.abspath(os.path.join(reg_path, "spim_template_to_ccf_syn_1Warp.nii.gz"))
    warp_ccf = ants.image_read(warp_file)
    
    return warp_template.orientation, warp_ccf.orientation
    

def orient_points(
        cells: list, orientation_in: str, orientation_out: str
) -> list:
    """
    Takes the cell locations and reorients them based on orientation
    parameters.
    
    axis 0: = z
    axis 1: = y
    axis 2: = x

    Parameters
    ----------
    cells : list
        cell locations
    reg_dims:
        the length of each axis of the state space the cells are in for when
        an axis needs to be flipped
    orientation_imaged : str
        the current orientation of the cells (i.e. spr)
    orientation_new : str
        the orientation that you want to conver the cells to (i.e. ras)

    Returns
    -------
    oriented_pts: list
        cell points in new orientation

    """
    
    _, swapped = get_orientation_transform(orientation_in, orientation_out)
    
    cell_array = np.array(cells)
    cells_out = cell_array[:, swapped]
    
    return cells_out

def read_ls_to_template_transform(reg_path: PathLike, reverse: False) -> tuple:
    """
    Imports ants transformation from raw image to smartspim template. These
    transforms are in "RAS" space

    Parameters
    -------------
    seg_path: PathLike
        Path to .gz file from registration

    Returns
    -------------
    ants.transform
        affine transform nonlinear warp field from ants.registration()
    """
    
    
    affine_file = os.path.abspath(os.path.join(reg_path, "ls_to_template_SyN_0GenericAffine.mat"))
    
    if reverse:
        warp_file = os.path.abspath(os.path.join(reg_path, "ls_to_template_SyN_1Warp.nii.gz"))
    else:
        warp_file = os.path.abspath(os.path.join(reg_path, "ls_to_template_SyN_1InverseWarp.nii.gz"))

    transforms = [
        warp_file,
        affine_file
    ]

    return transforms

def read_template_to_ccf_tramsform(reg_path: PathLike, reverse = False) -> tuple:
    """
    Imports ants static transformations from smartspim template to ccf. These
    transforms are in "ASL" space

    Parameters
    -------------
    seg_path: PathLike
        Path to .gz file from registration

    Returns
    -------------
    ants.transform
        affine transform nonlinear warp field from ants.registration()
    """
    
    affine_file = os.path.abspath(os.path.join(reg_path, "spim_template_to_ccf_syn_0GenericAffine.mat"))
    
    if reverse:
        warp_file = os.path.abspath(os.path.join(reg_path, "spim_template_to_ccf_syn_1Warp.nii.gz"))       
    else:
        warp_file = os.path.abspath(os.path.join(reg_path, "syn_1InverseWarp.nii.gz"))
    
    transforms = [
        warp_file,
        affine_file
    ]
    
    return transforms


def scale_cells(cells, scale):
    """
    Takes the downsampled cells, scales and orients them in smartspim template
    space.

    Parameters
    ----------
    cells : list
        list of cell locations that has been downsampled and oriented as
        [ML DV AP]
    scale : list
        the scaling metric between the raw image being downsampled to level 3
        and the image after being placed into 25um state space for

    Returns
    -------
    scaled_cells: list
        list of scaled cells that have been reoriented into the orientation of
        the template [ML, AP, DV]

    """
    
    scaled_cells = []
    for cell in cells:
        scaled_cells.append(
            (
                cell[0] * scale[0],
                cell[1] * scale[1],
                cell[2] * scale[2]
            )    
        )

    return scaled_cells

def cells_reformated(cells, save_path):
    
    cells_out = []
    for cell in cells:
        if all([x >= 1 for x in cell]):
            cells_out.append(
                Cell(
                    (cell[0], cell[1], cell[2]), 
                    1
                )    
            )
    
    
    save_cells(cells_out, save_path)
    
    return
    
def get_template_info(file_path: PathLike) -> dict:
    """
    

    Parameters
    ----------
    file_path : PathLike
        path to an nifti file that contains an ANTsImage template

    Returns
    -------
    params: dict
        information from file needed to convert cells into correct physical
        space

    """
    
    ants_img = ants.image_read(file_path)
    
    params = {
        'orientation': ants_img.orientation,
        'dims': ants_img.dimension,
        'scale': ants_img.spacing,
        'origin': ants_img.origin,
        'direction': ants_img.direction[
            np.where(ants_img.direction != 0)
        ]
    }
    
    return params

def convert_to_ants_space(template_parameters: dict, cells: np.ndarray):
    """
    Convert points from "index" space and places them into the physical space
    required for applying ants transforms for a given ANTsImage

    Parameters
    ----------
    template_parameters : dict
        parameters of the ANTsImage physical space that you are converting 
        the points
    cells : np.ndarray
        the location of cells in index space that have been oriented to the
        ANTs image that you are converting into

    Returns
    -------
    ants_pts : np.ndarray
        pts converted into ANTsPy physical space

    """
    
    ants_pts = cells.copy()
    
    for dim in range(template_parameters['dims']):
        ants_pts[:, dim] *= template_parameters['scale'][dim]
        ants_pts[:, dim] *= template_parameters['direction'][dim]
        ants_pts[:, dim] += template_parameters['origin'][dim]
        
    return ants_pts


def convert_from_ants_space(template_parameters: dict, cells: np.ndarray):
    """
    Convert points from the physical space of an ANTsImage and places 
    them into the "index" space required for visualizing

    Parameters
    ----------
    template_parameters : dict
        parameters of the ANTsImage physical space from where you are 
        converting the points
    cells : np.ndarray
        the location of cells in physical space 

    Returns
    -------
    pts : np.ndarray
        pts converted for ANTsPy physical space to "index" space

    """
    
    pts = cells.copy()
    
    for dim in range(template_parameters['dims']):
        pts[:, dim] -= template_parameters['origin'][dim]
        pts[:, dim] *= template_parameters['direction'][dim]
        pts[:, dim] /= template_parameters['scale'][dim]
        
    return pts

def apply_transforms_to_points(
        ants_pts: np.ndarray, transforms: list, invert: tuple) -> np.ndarray:
    """
    Takes the cell locations that have been converted into the correct
    physical space needed for the provided transforms and registers the points

    Parameters
    ----------
    ants_pts: np.ndarray
        array with cell locations placed into ants physical space
    transforms: list
        list of the file locations for the transformations

    Returns
    -------
    transformed_pts
        list of point locations in CCF state space

    """
    
    df = pd.DataFrame(ants_pts, columns=['x', 'y', 'z'])
    transformed_pts = ants.apply_transforms_to_points(
        3,
        df,
        transforms,
        whichtoinvert=invert
    )
    
    return np.array(transformed_pts)

def load_ccf_region(shared_path, region, split):
    
    mesh_dir = os.path.join(shared_path, "json_verts_float/")
    with open("{}{}.json".format(mesh_dir, region)) as f:
        structure_data = json.loads(f.read())
        vertices, faces = (
            np.array(structure_data[region]["vertices"]),
            np.array(structure_data[region]["faces"]),
        )
        
        if split == 'hemi':
            vertices_right = copy.copy(vertices)
            vertices_right[:, 0] = (
                vertices_right[:, 0] + (5700 - vertices_right[:, 0]) * 2
            )
        
            vertices = np.vstack((vertices, vertices_right))
        
    return vertices, faces


def get_volume(vertices, faces, split = False):
    
    if split:
        break_pt = len(vertices) // 2
        vert_L, vert_R = vertices[:break_pt], vertices[break_pt:]
        
        region_L = vedo.Mesh([vert_L, faces])
        region_R = vedo.Mesh([vert_R, faces])
        
        volume = region_L.volume() + region_R.volume()
    else:
        region = vedo.Mesh([vertices, faces])
        volume = region.volume()
        
    return volume

def get_mesh_interior_points(mesh):
    bounds = mesh.bounds()
    region_array = mesh.binarize(spacing = (1,1,1)).tonumpy()
    
    indecies = np.where(region_array == 255)
    xs = indecies[0] + int(bounds[0])
    ys = indecies[1] + int(bounds[2])
    zs = indecies[2] + int(bounds[4])
    
    return (xs, ys, zs)

def get_intensity_mask(verticies, faces, mask, split = 'hemi'):
    
    if split == 'hemi':
        break_pt = len(verticies) // 2
        vert_L, vert_R = verticies[:break_pt], verticies[break_pt:]
        
        region_L = vedo.Mesh([vert_L, faces])
        indicies = get_mesh_interior_points(region_L)
        mask[indicies]  = 1
        
        region_R = vedo.Mesh([vert_R, faces])
        indicies = get_mesh_interior_points(region_R)
        mask[indicies]  = 1
        
    else:
        region = vedo.Mesh([verticies, faces])
        indicies = get_mesh_interior_points(region)
        mask[indicies]  = 1
        
    return mask

def get_region_intensity(img, mask):  
    masked_img = np.where(mask > 0, img, 0)
    return masked_img
    
def mutual_information(hgram):
     """ 
     Mutual information for joint histogram
     """

     pxy = hgram / float(np.sum(hgram))
     px = np.sum(pxy, axis=1) # marginal for x over y
     py = np.sum(pxy, axis=0) # marginal for y over x
     px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
     # Now we can do the calculation using the pxy, px_py 2D arrays
     nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
     return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
 
def build_2d_histogram(img, ccf):
    
    img_pts = img.ravel()
    ccf_pts = ccf.ravel()
    
    bins = int(np.sqrt(len(img_pts / 5))) #This tensd to be a rule of thumb
    hist_2d, _, _ = np.histogram2d(img_pts, ccf_pts, bins=bins)
    
    return hist_2d

def crop_region(mask):
    
    x, y, z = np.where(mask > 0)
    
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)
    
    return [x_min, x_max, y_min, y_max, z_min, z_max]

def normalized_mutual_information(
    ccf_img: np.array, img: np.array, mask: np.array
) -> float:
    """
    Method to compute the mutual information error metric using numpy.
    Note: Check the used dtype to reach a higher precision in the metric

    See: Normalised Mutual Information of: A normalized entropy
    measure of 3-D medical image alignment,
    Studholme,  jhill & jhawkes (1998).

    Parameters
    ------------------------
    ccf_img: np.array
        2D/3D patch of extracted from the image 1
        and based on a windowed point.

    img: np.array
        2D/3D patch of extracted from the image 2
        and based on a windowed point.
        
    mask: np.array
        2D/

    Returns
    ------------------------
    float
        Float with the value of the mutual information error.
    """
    
    ccf_img = ccf_img.astype(int)

    
    # mutual information is invariant to scaling so this should not matter
    if img.dtype == np.dtype(np.float32):
        img = (img - img.min()) / (img.max() - img.min()) * ccf_img.max()
        img = img.astype(int)
    
    # get masks
    patch_1 = np.where(mask > 0, ccf_img, 0)
    patch_2 = np.where(mask > 0, img, 0)
    
    # flatten arrays
    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()

    # Compute the Normalized Mutual Information between the pixel distributions
    nmi = normalized_mutual_info_score(
        patch_1,
        patch_2,
        average_method='geometric'
    )
    
    return nmi

# code to transform cells from light sheet to ccf template based
def transform_cells(params):       
    """
    Takes cells and registers them to the CCF using the transforms from registration
    """
    
    img_path = f"{params['data_path']}/image_tile_fusing/OMEZarr/{params['channel']}.zarr"
      
    input_array = da.from_zarr(img_path, params['input_level']).squeeze()
    input_res = input_array.shape
    
    ds = 2**params['register_level']
    reg_dims = [int(dim / ds) for dim in input_res]
    
    print(reg_dims)
    
    raw_cells = read_xml(
        params['cells_path'], reg_dims, ds, params['orient'].lower(), params['institute_abbreviation']
    )
    
    _, swapped, mat = get_orientation_transform(params['orient'], 'ras')
    
    if params['orient'].lower() == 'rpi':
        scale = [14.4/25, 14.4/25, 16/25]
    else:
        scale = [16/25, 14.4/25, 14.4/25]

    scaled_cells = scale_cells(raw_cells, scale)
    
    orient_cells = np.array(scaled_cells)[:, swapped]
    
    #Transform to template
    if os.path.exists(f"{params['data_path']}/image_atlas_alignment/Ex_639_Em_660"):
        ls_transform_path = f"{params['data_path']}/image_atlas_alignment/Ex_639_Em_660"
    else:
        ls_transform_path = f"{params['data_path']}/image_atlas_alignment/Ex_639_Em_667"
    # ls_transform_path = f"{params['data_path']}/image_atlas_alignment/{params['reg_channel']}"
    
    transforms = read_ls_to_template_transform(ls_transform_path, reverse = False)
    template_params = get_template_info('../data/lightsheet_template_ccf_registration/smartspim_lca_template_25.nii.gz')
    ants_pts = convert_to_ants_space(template_params, orient_cells)
    template_pts = apply_transforms_to_points(ants_pts, transforms, invert = (False, True))
    
    template_cells = convert_from_ants_space(template_params, template_pts)
    
    transforms = read_template_to_ccf_tramsform(params['transforms_path'], reverse = False)
    ccf_params = get_template_info('../data/lightsheet_template_ccf_registration/ccf_average_template_25.nii.gz')
    ccf_pts = apply_transforms_to_points(template_pts, transforms, invert = (False, True))
    out = convert_from_ants_space(ccf_params, ccf_pts)
    
    _, swapped, _ = get_orientation_transform('RAS', 'ASL')
    ccf_cells = out[:, swapped]

    return template_cells, ccf_cells
    
    

    
