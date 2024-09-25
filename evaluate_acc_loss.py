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
        [[25, 128]],
        # [[50, 128]],
        # [[75, 128]],
        # [[100, 128]],
        # [[125, 128]],
        # [[150, 128]],
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
    out = torch.randn(25, 128).float()
    # change out to numpy array
    out = out.numpy()

    # Save 'out' as a npy file
    np.save('evaluate_data.npy', out)

    # print(score)
    # Export RKNN model
    print('--> Evaluate RKNN model')
    ret = rknn.accuracy_analysis(inputs=['evaluate_data.npy'], target='rk3588')
    print(ret)

    # Release RKNN object for next use
    rknn.release()

if __name__ == "__main__":

    # Define the base path for models
    base_model_path = "./models/lossAV"

    # Use the base path to construct the full paths
    scripted_model_path = f"{base_model_path}.pt"
    onnx_model_path = f"{base_model_path}.onnx"
    rknn_model_path = f"{base_model_path}.rknn"

    evaluate(onnx_model_path, rknn_model_path)

# I rknn-toolkit2 version: 2.0.0b0+9bab5682
# --> Config model
# W config: Please make sure the model can be dynamic when enable 'config.dynamic_input'!
# I The 'dynamic_input' function has been enabled, the MaxShape is dynamic_input[0] = [[25, 128]]!
#           The following functions are subject to the MaxShape:
#             1. The quantified dataset needs to be configured according to MaxShape
#             2. The eval_perf or eval_memory return the results of MaxShape
# done
# --> Loading ONNX model
# I It is recommended onnx opset 19, but your onnx model opset is 12!
# I Model converted from pytorch, 'opset_version' should be set 19 in torch.onnx.export for successful convert!
# I Loading : 100%|██████████████████████████████████████████████████| 4/4 [00:00<00:00, 19784.45it/s]
# W load_onnx: The config.mean_values is None, zeros will be set for input 0!
# W load_onnx: The config.std_values is None, ones will be set for input 0!
# done
# --> Building model
# W build: Can not find 'idx' to insert, default insert to 0!
# I rknn building ...
# I rknn buiding done.
# done
# --> Evaluate RKNN model
# adb: unable to connect for root: closed
# I target set by user is: rk3588
# I Get hardware info: target_platform = rk3588, os = Linux, aarch = aarch64
# I Check RK3588 board npu runtime version
# I Starting ntp or adb, target is RK3588
# I Start adb...
# I Connect to Device success!
# I NPUTransfer: Starting NPU Transfer Client, Transfer version 2.1.0 (b5861e7@2020-11-23T11:50:36)
# D RKNNAPI: ==============================================
# D RKNNAPI: RKNN VERSION:
# D RKNNAPI:   API: 2.0.0b0 (18eacd0 build@2024-03-22T06:07:59)
# D RKNNAPI:   DRV: rknn_server: 1.5.2 (8babfea build@2023-08-25T10:30:31)
# D RKNNAPI:   DRV: rknnrt: 1.5.2 (c6b7b351a@2023-08-23T15:28:22)
# D RKNNAPI: ==============================================
# D RKNNAPI: Input tensors:
# D RKNNAPI:   index=0, name=input, n_dims=2, max dims=[25, 128], n_elems=3200, size=6400, w_stride = 0, size_with_stride = 0, fmt=UNDEFINED, type=FP16, qnt_type=NONE, zp=0, scale=1.000000
# D RKNNAPI: Output tensors:
# D RKNNAPI:   index=0, name=output, n_dims=1, max dims=[25], n_elems=25, size=50, w_stride = 0, size_with_stride = 0, fmt=UNDEFINED, type=FP16, qnt_type=NONE, zp=0, scale=1.000000
# adb: unable to connect for root: closed
# /userdata/dumps/: 6 files pulled. 0.1 MB/s (26580 bytes in 0.365s)
# I Save Tensors to txt: 100%|█████████████████████████████████████████| 6/6 [00:00<00:00, 194.34it/s]
# I GraphPreparing : 100%|████████████████████████████████████████████| 4/4 [00:00<00:00, 1035.57it/s]
# I AccuracyAnalysing : 100%|███████████████████████████████████████████| 4/4 [00:00<00:00, 23.75it/s]

# # simulator_error: calculate the output error of each layer of the simulator (compared to the 'golden' value).
# #              entire: output error of each layer between 'golden' and 'simulator', these errors will accumulate layer by layer.
# #              single: single-layer output error between 'golden' and 'simulator', can better reflect the single-layer accuracy of the simulator.
# # runtime_error: calculate the output error of each layer of the runtime.
# #              entire: output error of each layer between 'golden' and 'runtime', these errors will accumulate layer by layer.
# #              single_sim: single-layer output error between 'simulator' and 'runtime', can better reflect the single-layer accuracy of runtime.

# layer_name                                            simulator_error                             runtime_error                      
#                                                   entire              single                  entire           single_sim            
#                                                cos      euc        cos      euc            cos      euc        cos      euc          
# -----------------------------------------------------------------------------------------------------------------------------
# [Input] input                                1.00000 | 0.0       1.00000 | 0.0           1.00000 | 0.0118    1.00000 | 0.0118        
# [Reshape] /FC/Gemm_2conv_reshape1_output     1.00000 | 0.0118    1.00000 | 0.0118        1.00000 | 0.0118    1.00000 | 0.0           
# [Conv] /FC/Gemm_output_0_conv                1.00000 | 0.0034    1.00000 | 0.0034        1.00000 | 0.0034    1.00000 | 0.0           
# [Slice] output_slice_shape4                  1.00000 | 0.0023    1.00000 | 0.0014        1.00000 | 0.0023    1.00000 | 0.0           
# [Reshape] output                             1.00000 | 0.0023    1.00000 | 0.0014        1.00000 | 0.0023    1.00000 | 0.0           
# I The error analysis results save to: ./snapshot/error_analysis.txt
# W accuracy_analysis: The mapping of layer_name & file_name save to: ./snapshot/map_name_to_file.txt
# 0