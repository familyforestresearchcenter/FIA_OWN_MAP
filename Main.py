#!/usr/bin/env python
# coding: utf-8

import os
import sys
from zipfile import ZipFile
import zipfile

sys.path.insert(0, os.path.abspath('.'))

from utils.helpers import *
from Name_Matching_2_8 import *
from Classify_Unknowns import *
# from Land_Analysis import *
# from Summary_Script import *
import shutil

if __name__ == '__main__':
    
    t1 = time.time()
    TMP_DIR = '/dev/shm'
    # Old Version

    # exec(open("Preprocessing.py").read())    
    # exec(open("Classify_Unknowns.py").read())
    # exec(open("Name_Matching_2_8.py").read())
    # exec(open("Summary_Script.py").read())
    # exec(open("New_Map_Data.py").read())
    # exec(open("Last_Overlay_v2.py").read())
    # state_name = [i for i in os.listdir('./data') if len(i) == 2][0]
    # shutil.make_archive(f'./data/{state_name}', 'zip', f'./data/{state_name}')
    # shutil.rmtree(f'./data/{state_name}')
    # t2 = time.time()
    # print(f'{state_name} Completed in : {t2-t1} s\n')

    # new version off the tmp folder
    # === Run pipeline steps
    exec(open("Preprocessing_opt.py").read())

    exec(open("Classify_Unknowns_opt.py").read())
    # NOW we can safely delete temp.csv
    temp_csv = os.path.join(TMP_DIR, 'temp.csv')
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    exec(open("Name_Matching_opt.py").read())
    # NOW we can safely delete new_classified_state_temp.csv
    classified_csv = os.path.join(TMP_DIR, 'new_classified_state_temp.csv')
    if os.path.exists(classified_csv):
        os.remove(classified_csv)

    exec(open("Summary_Script_opt.py").read())

    exec(open("Map_Data_opt.py").read())

    exec(open("Last_Overlay_opt.py").read())
    # NOW we can delete encoded raster
    encoded_tif = os.path.join(TMP_DIR, 'Own_Type_by_Landcover_Encoded.tif')
    if os.path.exists(encoded_tif):
        os.remove(encoded_tif)


    # === 1. Get state_name from GDB in /tmp/
    gdb_file = next((f for f in os.listdir(TMP_DIR) if f.endswith('.gdb')), None)
    if gdb_file is None:
        raise FileNotFoundError("No .gdb file found in /dev/shm/")
    state_name = gdb_file.split('.')[0]

    # === 2. Rename Full and Reduced tables with state prefix
    full_table_old = os.path.join(TMP_DIR, 'Full_Data_Table.csv')
    reduced_table_old = os.path.join(TMP_DIR, 'Reduced_Data_Table.csv')
    final_raster_path = os.path.join(TMP_DIR, f'{state_name}_Final_Encoded.tif')

    full_table_new = os.path.join(TMP_DIR, f'{state_name}_Full_Data_Table.csv')
    reduced_table_new = os.path.join(TMP_DIR, f'{state_name}_Reduced_Data_Table.csv')
    parcel_id_path = os.path.join(TMP_DIR, f"{state_name}_Parcel_IDs.json")

    os.rename(full_table_old, full_table_new)
    os.rename(reduced_table_old, reduced_table_new)

    # === 3. Create zip as /tmp/{state_name}.zip
    zip_path = os.path.join(TMP_DIR, f'{state_name}.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(full_table_new, arcname=os.path.basename(full_table_new))
        # zipf.write(reduced_table_new, arcname=os.path.basename(reduced_table_new))
        zipf.write(final_raster_path, arcname=os.path.basename(final_raster_path))
        zipf.write(parcel_id_path, arcname=os.path.basename(parcel_id_path))

    # # # ✅ Include the Parcel_IDs json
    #     parcel_id_path = os.path.join(TMP_DIR, f"{state_name}_Parcel_IDs.json")
    #     if os.path.exists(parcel_id_path):
    #         zipf.write(parcel_id_path, arcname=os.path.basename(parcel_raster_path))


    print(f"\n✅ Zipped outputs written to: {zip_path}")

    # Final cleanup, keep only WA.zip and system-private dirs
    files_to_keep = {f"{state_name}.zip"}
    system_dirs_prefixes = ("systemd-private-", "snap-private-tmp")

    for fname in os.listdir(TMP_DIR):
        fpath = os.path.join(TMP_DIR, fname)

        # Skip the zip and system/private dirs
        if fname in files_to_keep or fname.startswith(system_dirs_prefixes):
            continue

        try:
            if os.path.isdir(fpath):
                shutil.rmtree(fpath, ignore_errors=True)
            else:
                os.remove(fpath)
        except Exception as e:
            print(f"Warning: could not delete {fpath} — {e}")


    # === 4. Report runtime
    t2 = time.time()
    print(f'{state_name} Completed in : {t2 - t1:.2f} seconds\n')


