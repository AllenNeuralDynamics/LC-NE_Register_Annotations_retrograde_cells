
import os
import re
import vedo
import json

import numpy as np
import pandas as pd
import point_cloud_utils as pcu

from glob import glob
from datetime import datetime
from natsort import natsorted
from utils import utils, json_to_xml
from utils import pipeline_utils as pu

import aind_data_schema.core.processing as ps
import aind_data_schema.core.data_description as ds
from aind_data_schema.core.metadata import Metadata
from aind_data_access_api.document_db import MetadataDbClient

params = {
    # data collection
    'data_folder': '../data',
    'annotation_folder': 'annotations',
    'tp_name': 'PK_LC_all_annotations',
    'json_path': '../results/jsons',
    'csv_path': '../results/raw_space',

    # shared
    'mesh_path': '../code/ccf_files/CCF_meshes/json_verts_float',
    'mesh_ids': [771, 997],
    'ccf_res': 25,
    'ccf_template': '../data/SmartSPIM-template_2024-05-16_11-26-14/ccf_average_template_25.nii.gz',
    'smartspim_template': '../data/SmartSPIM-template_2024-05-16_11-26-14/smartspim_lca_template_25.nii.gz',
    'template_res': [14.4, 14.4, 16],
    'resolution': 5_000,
    'threshold': 300,

    # mesh warping
    'ccf_transforms': [
        '../data/SmartSPIM-template_2024-05-16_11-26-14/spim_template_to_ccf_syn_1Warp_25.nii.gz',
        '../data/SmartSPIM-template_2024-05-16_11-26-14/spim_template_to_ccf_syn_0GenericAffine_25.mat'
    ],

    # registration
    'reg_save_path': '../results/registered',
    'template_to_ccf': [
        '../data/SmartSPIM-template_2024-05-16_11-26-14/spim_template_to_ccf_syn_0GenericAffine_25.mat',
        '../data/SmartSPIM-template_2024-05-16_11-26-14/spim_template_to_ccf_syn_1InverseWarp_25.nii.gz'
    ],
    'downsample': 3,
    'scaling': [16 / 25, 14.4 / 25, 14.4 / 25],
    'institute': 'AIND',

    # signed distance final
    'sd_save_path': '../results/final_results',
}


def collect_datasets(params):
    folders = glob(os.path.join(params['data_folder'], 'SmartSPIM_*'))
    data = {}
    for folder in folders:
        lt_id = folder.split('_')[1]

        register_folder = natsorted(glob(os.path.join(folder, 'image_atlas_alignment', 'Ex_*_Em_*')))[-1]
        template_transforms = [
            f"{register_folder}/ls_to_template_SyN_1Warp.nii.gz",
            f"{register_folder}/ls_to_template_SyN_0GenericAffine.mat"
        ]

        res_path = natsorted(glob(os.path.join(folder, 'image_tile_fusing', 'OMEZarr', 'Ex_*_Em_*.zarr/0/.zarray')))[-1]
        input_res = json_to_xml.read_json_as_dict(res_path)["shape"]

        input_dims = [
            input_res[-1],
            input_res[-2],
            input_res[-3],
        ]

        try:
            annotation_file = glob(os.path.join(params['data_folder'], params['annotation_folder'], f'{lt_id}*.json'))[0]
            annotation_data = json_to_xml.read_json_as_dict(annotation_file)
        except:
            print(f"Need to add annotation JSON for dataset {lt_id}.")
            continue

        cells = []
        for layers in annotation_data['layers']:
            if isinstance(layers['source'], dict):
                for cell in layers['annotations']:
                    cells.append(
                        [
                            cell['point'][0],
                            cell['point'][1],
                            cell['point'][2]
                        ]
                    )

        acquisition = json_to_xml.read_json_as_dict(os.path.join(folder, 'acquisition.json'))

        data[lt_id] = {
            "template_transforms": template_transforms,
            "annotation_file": annotation_file,
            "orientation": acquisition['axes'],
            "input_dims": input_dims,
            "cell_locations": cells,
            "ng_file": annotation_data,
        }

    print(f"Will be processing {len(data)} datasets")
    return data


def process_meshes(params, data):
    if not os.path.exists(params['json_path']):
        os.mkdir(params['json_path'])

    if not os.path.exists(params['csv_path']):
        os.mkdir(params['csv_path'])

    for lt_id, dataset in data.items():

        meshes = {}

        for struct in params['mesh_ids']:

            verts, faces = pu.load_json_mesh(params['mesh_path'], str(struct))
            verts = verts[:, [0, 2, 1]] / params['ccf_res']

            verts_ls, faces_ls = pu.warp_mesh(
                verts,
                faces,
                params['ccf_template'],
                params['smartspim_template'],
                params['ccf_transforms'],
                dataset['template_transforms']
            )

            scale = [params['ccf_res']/dim for dim in params['template_res']]
            for i in range(3):
                verts_ls[:, i] = verts_ls[:, i] * scale[i]

            verts_ls *= 2**3

            orientation = utils.get_orientation(dataset['orientation'])
            _, swapped, mat = utils.get_orientation_transform('ras', orientation)

            verts_orient = pu.orient_mesh(
                vert_list=verts_ls,
                reg_dims=data[lt_id]['input_dims'],
                ds=1,
                orient='ras',
                orient_matrix=mat,
                institute='AIND',
            )

            verts_orient = verts_orient[:, swapped]

            if struct != 997:

                print(f'Getting cells for {struct} in dataset {lt_id}')

                vw, fw = pcu.make_mesh_watertight(verts_orient, faces_ls, params['resolution'])

                pts = np.array(dataset['cell_locations'], dtype = np.double)
                sdf, _, _ = pcu.signed_distance_to_mesh(pts, vw, fw)

                sd_points = []
                for c, s in enumerate(sdf):
                    if s < 0:
                        sd_points.append([pts[c, 0], pts[c, 1], pts[c, 2], 'inside'])
                    elif s < params['threshold']:
                        sd_points.append([pts[c, 0], pts[c, 1], pts[c, 2], 'boarder'])
                    else:
                        sd_points.append([pts[c, 0], pts[c, 1], pts[c, 2], 'outside'])

                meshes[struct] = {
                    'faces': fw,
                    'verts': vw
                }

                meshes[struct]['cells'] = sd_points


                df = pd.DataFrame(sd_points, columns = ['Z', 'Y', 'X', 'Location'])
                df.to_csv(f"{params['csv_path']}/{str(lt_id)}_cells.csv")

            else:
                meshes[struct] = {
                    'faces': faces_ls,
                    'verts': verts_orient
                }

        data[lt_id]['meshes'] = meshes


def register_cells(params, data):
    registered_cells = {}
    ccf_params = None

    for file in glob('../results/raw_space/*.csv'):

        lt_id = re.findall('[0-9]{6}', file)[0]
        print(f'registering dataset: {lt_id}')

        ls_to_template = [
            glob(f'../data/SmartSPIM_{lt_id}*/image_atlas_alignment/**/ls_to_template_SyN_0GenericAffine.mat')[0],
            glob(f'../data/SmartSPIM_{lt_id}*/image_atlas_alignment/**/ls_to_template_SyN_1InverseWarp.nii.gz')[0]
        ]

        print(f'lightsheet to template transforms: {ls_to_template}')

        acquisition_file = glob(f'../data/SmartSPIM_{lt_id}*/acquisition.json')[0]
        zarr_path = glob(f'../data/SmartSPIM_{lt_id}*/image_tile_fusing/OMEZarr/*.zarr')[0]

        res_metadata = f"{zarr_path}/0/.zarray"
    
        input_res = utils.read_json_as_dict(res_metadata)["shape"]

        # input res is returned in order tczyx, here we use xzy
        # where z is the imaging axis
        res = [
            input_res[-3],
            input_res[-2],
            input_res[-1],
        ]

        # get dimensions at registered scale
        ds = 2**params['downsample']
        reg_dims = [dim / ds for dim in res]

        # get orientation information
        orientation = utils.read_json_as_dict(acquisition_file)
        orient = utils.get_orientation(orientation['axes'])
        template_params = utils.get_template_info(params["smartspim_template"])
    
        _, swapped, mat = utils.get_orientation_transform(
            orient, 
            template_params["orientation"]
        )
    
        raw_cells = utils.read_cells_from_csv(
            cell_likelihoods_path=file,
            reg_dims=reg_dims,
            ds=ds,
            orient=orient,
            orient_matrix=mat,
            institute=params['institute'],
        )

        scaled_cells = utils.scale_cells(raw_cells, params['scaling'])
        scaled_cells = np.array(scaled_cells)
        orient_cells = scaled_cells[:, swapped]

        #Converting oriented cells into ANTs physical space
        template_params = utils.get_template_info(params['smartspim_template'])
        ants_cells = utils.convert_to_ants_space(template_params, orient_cells)

        #Registering Cells to SmartSPIM template
        template_cells = utils.apply_transforms_to_points(
            ants_cells, ls_to_template, invert=(True, False)
        )

        #Convert template cells into CCF space and orientation
        ccf_pts = utils.apply_transforms_to_points(
            template_cells, params['template_to_ccf'], invert=(True, False)
        )

        #Convert cells back into index space
        ccf_params = utils.get_template_info(params['ccf_template'])
        ccf_cells = utils.convert_from_ants_space(ccf_params, ccf_pts)

        _, swapped, _ = utils.get_orientation_transform(
            template_params["orientation"], ccf_params["orientation"]
        )

        cells_transformed = ccf_cells[:, swapped]

        # Save the transformed coordinates for this lt_id
        pu.save_coordinates_with_indices_to_csv(cells_transformed, lt_id, output_dir=params['reg_save_path'])

        registered_cells[lt_id] = cells_transformed

    return registered_cells, ccf_params


def load_ccf_meshes(params):
    mesh_dict = {}

    for mesh_id in params['mesh_ids']:
        verts, faces = pu.load_json_mesh(params['mesh_path'], str(mesh_id))
        mesh_dict[str(mesh_id)] = {
            "verts": verts,
            "faces": faces
        }

    return mesh_dict


def compute_signed_distances(params, mesh_dict, ccf_params, registered_cells):
    """
    Need to orient points to match the mesh files


    Orientation of CCF Template (points out of registration)

    X: AP (Anterior -> Posterior)

    Y: DV (Superior -> Inferior)

    Z: ML (Left -> Right)


    Orientation of mesh

    X: ML (Right -> Left)

    Y: DV (Superior -> Inferior)

    Z: AP (Anterior -> Posterior)
    """
    if not os.path.exists(params['sd_save_path']):
        os.mkdir(params['sd_save_path'])

    ccf_orientation = ccf_params["orientation"]
    print(ccf_orientation)
    mesh_orientation = 'RSA'

    _, swapped, _ = utils.get_orientation_transform(
        ccf_orientation, mesh_orientation
    )

    print(swapped)

    mesh_id = '771'
    v, f = mesh_dict[str(mesh_id)]['verts'], mesh_dict[str(mesh_id)]['faces']
    vw, fw = pcu.make_mesh_watertight(v, f, params['resolution'])

    for lt_id, pts in registered_cells.items():
        pts = pts[:, swapped] * params['ccf_res']
        pts = np.array(pts, dtype = np.double, order = 'F')
        vw = np.array(vw, order = 'F')
        sdf, _, _= pcu.signed_distance_to_mesh(pts, vw, fw)

        pts /= params['ccf_res']

        sd_points = []
        for c, s in enumerate(sdf):
            if s < 0:
                sd_points.append([pts[c, 0], pts[c, 1], pts[c, 2], 'inside'])
            elif s < params['threshold']:
                sd_points.append([pts[c, 0], pts[c, 1], pts[c, 2], 'boarder'])
            else:
                sd_points.append([pts[c, 0], pts[c, 1], pts[c, 2], 'outside'])

        points_df = pd.DataFrame(sd_points, columns = ['ML', 'DV', 'AP', 'Location'])

        filename = f"{lt_id}_registered_pts.csv"
        points_df.to_csv(f"{params['sd_save_path']}/{filename}")

        print(f"Saving data for dataset {lt_id}: {filename}\n")

def create_intermediate_asset():
    return

def main():
    start_time = datetime.now()
    data = collect_datasets(params)
    process_meshes(params, data)

    registered_cells, ccf_params = register_cells(params, data)
    mesh_dict = load_ccf_meshes(params)
    compute_signed_distances(params, mesh_dict, ccf_params, registered_cells)

    creation_time = datetime.now()
    base_name = "LC-NE-Register-Annotations-retrograde-cells"

    name = ds.build_data_name(base_name, creation_time)

    code_details = ps.Code(
        name=name,
        url="https://github.com/AllenNeuralDynamics/LC-NE_Register_Annotations_retrograde_cells.git",
        version="2.0",
        parameters=params,
        input_data=[
            ps.DataAsset(name=folder) for folder in glob(os.path.join(params['data_folder'], 'SmartSPIM*'))
        ]
    )

    end_time = datetime.now()
    my_processing = ps.Processing(
        data_processes=[
            ps.DataProcess(
                stage=ps.ProcessStage.PROCESSING,
                process_type=ps.ProcessName.OTHER,
                name=name,
                experimenters=["Polina Kosillo"],
                start_date_time=start_time,
                end_date_time=end_time,
                code=code_details,
                notes="Cell locations from manually inspected calssification are registered to the CCF and location is identified as within or outside the Pons"
            ),
        ],
    )

    new_dd = ds.DataDescription(
        name=name,
        source_data=[
            str(folder.split('/')[-1]) for folder in glob(os.path.join(params['data_folder'], 'SmartSPIM_*'))
        ],
        creation_time=creation_time,
        institution=ds.Organization.AIND,
        data_level=ds.DataLevel.DERIVED,
        investigators=[ds.Person(name="Polina Kosillo")],
        project_name="Discovery Neuromodulation - Subproject 2 Molecular Anatomy Cell Types",
        modalities=[ds.Modality.SPIM],
        license=ds.License.CC_BY_40,
        funding_source=[
            ds.Funding(funder=ds.Organization.NIMH, grant_number="1R01MH134833"),
            ds.Funding(funder=ds.Organization.AI)
        ],
        data_summary="Cell locations from manually inspected calssification are registered to the CCF and location is identified as within, outside, or boardering the Pons"
    )

    new_md = Metadata(
        name=name,
        data_description=new_dd,
        processing=my_processing,
        location='../results'
    )
    
    output_path = "../results"
    new_md.data_description.write_standard_file(output_path)
    new_md.processing.write_standard_file(output_path)

    print("/n/n################# Processing Completed #################")


if __name__ == '__main__':
    main()
