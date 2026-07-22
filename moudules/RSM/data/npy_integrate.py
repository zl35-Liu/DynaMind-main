

























import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

base_path = "/path/to/cinebrain/dataset/"
































def process_single_concat(args):
    """
    Concatenate multiple NPY arrays along axis one for a single task. When axis zero contains 70 channels, retain only the first 69.
    """
    j, data_path, output_path = args
    try:

        file_paths = [f"{data_path}{j * 5 + k}.npy" for k in range(5)]


        arrays_to_concat = []

        for file_path in file_paths:

            npy_part = np.load(file_path)



            if npy_part.shape[0] == 70:


                npy_part = npy_part[:69, ...]


            arrays_to_concat.append(npy_part)




        if arrays_to_concat:
            npy_cat = np.concatenate(arrays_to_concat, axis=1)
        else:

            print(f"No files found for index {j}")
            return False



        np.save(f"{output_path}{j}.npy", npy_cat)
        return True

    except Exception as e:

        print(f"\nError processing index {j}: {e}")
        return False

def process_subject_parallel(subject_idx, max_workers=None):
    """
    Process one subject in parallel.
    """
    data_path = base_path + f"sub{subject_idx}/eeg_02/"
    output_path = base_path + f"sub{subject_idx}/eeg_cat/"
    os.makedirs(output_path, exist_ok=True)

    print(f"Processing subject {subject_idx}...")


    tasks = [(j, data_path, output_path) for j in range(8100)]


    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        futures = {executor.submit(process_single_concat, task): task[0]
                   for task in tasks}


        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()

    print(f"Subject {subject_idx} processing complete.")


if __name__ == '__main__':
    for i in range(1):
        process_subject_parallel(i+1, max_workers=4)