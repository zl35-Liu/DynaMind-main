import numpy as np
import mne
from mne.preprocessing import ICA



SFREQ = 1000

N_EEG = 64

N_ECG = 5

NOTCH_FREQ = 50.0


ch_names = [f'EEG {i:02d}' for i in range(N_EEG)] + \
           [f'ECG {i:02d}' for i in range(N_ECG)]
ch_types = ['eeg'] * N_EEG + ['ecg'] * N_ECG


ECG_REF_CH_NAME = ch_names[N_EEG]



def preprocess_eeg_data(raw_data_array: np.ndarray) -> np.ndarray:
    """
    Preprocess a raw EEG or ECG array shaped (69, 4000), apply filtering and ICA, and return cleaned EEG data shaped (64, 4000). The input is assumed to use volts.
    """
    print("--- Step 1: initialize the MNE data structure ---")
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)
    raw = mne.io.RawArray(raw_data_array, info, verbose=False)






    raw_eeg = raw.copy().pick_types(eeg=True)
    print(f"Raw data shape: {raw_data_array.shape}")
    print(f"The processing object contains  {raw_eeg.info['nchan']}  EEG channels.")



    print("\n--- Step 2: filtering (0.1-30 Hz band-pass and 50 Hz notch) ---")


    raw_eeg.filter(l_freq=0.1, h_freq=30.0, fir_design='firwin', verbose=False)
    print("  -> Success: band-pass filtering (0.1-30.0 Hz)")


    raw_eeg.notch_filter(NOTCH_FREQ, fir_design='firwin', verbose=False)
    print(f"  -> Success: notch filtering ({NOTCH_FREQ} Hz)")




    print("\n--- Step 3: independent component analysis (ICA) ---")



    ica = ICA(n_components=N_EEG, method='fastica', random_state=42)


    ica.fit(raw_eeg)
    print("  -> Success: ICA fitting completed.")



    ecg_indices, ecg_scores = ica.find_bads_ecg(
        raw,
        ch_name=ECG_REF_CH_NAME,

        verbose=False
    )
    print(f"  -> Automatically detected ECG artifact component indices: {ecg_indices}")


    ica.exclude = ecg_indices






    print(f"  -> Excluding  {len(ica.exclude)}  components and reconstructing EEG data...")
    ica.apply(raw_eeg, verbose=False)
    print("  -> Success: EEG reconstruction completed.")



    clean_eeg_data = raw_eeg.get_data()
    print(f"\nFinal clean EEG shape: {clean_eeg_data.shape}")

    return clean_eeg_data



if __name__ == '__main__':


    try:





        raw_data = np.random.randn(69, 4000) * 1e-5
        print(f"Simulated raw data loaded: {raw_data.shape}")


        clean_eeg_segment = preprocess_eeg_data(raw_data)




    except FileNotFoundError:
        print("Error: replace the NPY path in the script and ensure the file exists.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")