from rknn.api import RKNN
import sys

# Execute 
# sudo adbd &
# to start adb server on the rk3588s board

# Function to generate RKNN model
def evaluate_rknn_model(rknn_path):
    # Create RKNN object
    rknn = RKNN()

    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform='rk3588', optimization_level=3)
    print('done')
    
    rknn.load_rknn(rknn_path)

    ret = rknn.init_runtime(target='rk3588', perf_debug=True)

    # print(score)
    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.eval_perf(fix_freq=True)
    print(ret)

    # Release RKNN object for next use
    rknn.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        # Define the base path for models
        base_model_path = "./models/lightASD"

        rknn_model_path = f"{base_model_path}.rknn"
    else:
        # Get the model path from the command line argument
        rknn_model_path = sys.argv[1]

    evaluate_rknn_model(rknn_model_path)
