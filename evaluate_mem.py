import torch
from rknn.api import RKNN

# Execute 
# sudo adbd &
# to start adb server on the rk3588s board

# Function to generate RKNN model
def evaluate_rknn_model(rknn_path):
    # Create RKNN object
    rknn = RKNN()

    dynamic_input = [
        [[25, 128]],
    ]

    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform='rk3588', optimization_level=3)
    # rknn.config(target_platform='rk3588', optimization_level=3, dynamic_input=dynamic_input)
    print('done')
    
    rknn.load_rknn(rknn_path)

    # ret = rknn.init_runtime(target='rk3588', perf_debug=True)
    ret = rknn.init_runtime(target='rk3588', eval_mem=True)

    # print(score)
    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.eval_memory()
    print(ret)

    # Release RKNN object for next use
    rknn.release()

if __name__ == "__main__":

    # Define the base path for models
    base_model_path = "./models/lightASD"

    # Use the base path to construct the full paths
    rknn_model_path = f"{base_model_path}.rknn"

    evaluate_rknn_model(rknn_model_path)
