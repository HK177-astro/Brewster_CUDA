import os
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"  # 1: use the CUDA Simulator 0: use GPUs
from src.brewster.setup import setup
import pickle
import pytest

@pytest.fixture(scope="session", autouse=True)
def inputs():
    with open("src/tests/test_data/forward_model_input_R10K_water.pic", "rb") as f:
        inputs = pickle.load(f, encoding="bytes")

    gaslistpath = "src/tests/test_data/gaslist.dat"
    
    grav, tot_VMR_molmass, mu, fH, fHmin, fH2, fHe, nlayers, press, temp = setup(inputs=inputs, gaslistpath=gaslistpath)

    input_dict = {
        "grav": grav,
        "tot_VMR_molmass": tot_VMR_molmass,
        "mu": mu,
        "fH": fH,
        "fHmin": fHmin,
        "fH2": fH2,
        "fHe": fHe,
        "nlayers": nlayers,
        "press": press,
        "temp": temp
    }

    return input_dict


@pytest.fixture(scope="session", autouse=True)
def kernel_config(inputs):
    # Define block and grid dimensions
    threads_per_block = 64
    blocks_per_grid = (inputs['nlayers'] + threads_per_block - 1) // threads_per_block

    return threads_per_block, blocks_per_grid