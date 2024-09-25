import torch
from rknn.api import RKNN
import numpy as np

# Execute 
# sudo adbd &
# to start adb server on the rk3588s board

# Function to generate RKNN model
def evaluate(onnx_path, rknn_path):
    # Create RKNN object
    rknn = RKNN()

    dynamic_input = [
        [[1, 100, 13], [1, 25, 112, 112]],
        # [[1, 200, 13], [1, 50, 112, 112]],
        # [[1, 300, 13], [1, 75, 112, 112]],
        # [[1, 400, 13], [1, 100, 112, 112]],
        # [[1, 500, 13], [1, 125, 112, 112]],
        # [[1, 600, 13], [1, 150, 112, 112]]
    ]

    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform='rk3588', optimization_level=3, dynamic_input=dynamic_input)
    print('done')

    print('--> Loading ONNX model')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('Load ONNX model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build RKNN model failed!')
        exit(ret)
    print('done')

    # Generate test data and save in npy file
    out = torch.randn(1, 100, 13).float()
    # change out to numpy array
    out = out.numpy()

    # Save 'out' as a npy file
    np.save('evaluate_data_1.npy', out)

    out = torch.randn(1, 25, 112, 112).float()
    # change out to numpy array
    out = out.numpy()

    # Save 'out' as a npy file
    np.save('evaluate_data_2.npy', out)

    # print(score)
    # Export RKNN model
    print('--> Evaluate RKNN model')
    ret = rknn.accuracy_analysis(inputs=['evaluate_data_1.npy', 'evaluate_data_2.npy'], target='rk3588')
    print(ret)

    # Release RKNN object for next use
    rknn.release()

if __name__ == "__main__":

    # Define the base path for models
    base_model_path = "./models/lightASD"

    # Use the base path to construct the full paths
    scripted_model_path = f"{base_model_path}.pt"
    onnx_model_path = f"{base_model_path}.onnx"
    rknn_model_path = f"{base_model_path}.rknn"

    evaluate(onnx_model_path, rknn_model_path)
