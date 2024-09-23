import torch
from ASD import ASD
from rknn.api import RKNN

# Need pytorch 2.4.0

# Function to create TorchScript model
def create_torchscript_model(asd_model_path, save_path):
    # Load pretrained ASD model
    asd_model = ASD()
    asd_model.loadParameters(asd_model_path)
    asd_model.model.eval()
    asd_model.lossAV.eval()
    score_example = torch.rand([25, 128])
    try:
        # Trace the model ensuring the input is a tensor wrapped in a tuple
        lossAV_model = torch.jit.trace(asd_model.lossAV, (score_example,), strict=False)
        
        # Save the traced model to a file named "lossAV.pt"
        torch.jit.save(lossAV_model, save_path)
        print(f"Model {save_path} successfully traced and saved.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to export ONNX model from TorchScript model
def export_onnx_model(script_model_path, onnx_path):
    # Load the TorchScript model
    torch_script_loss_model = torch.jit.load(script_model_path)

    # Loss need to be exported under pytorch 2.4.0? in my dev env, need to be exported out of the venv
    # Export to ONNX
    dummy_loss = torch.randn(25, 128).float()

    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'},
    }

    torch.onnx.export(
        torch_script_loss_model,
        dummy_loss,
        onnx_path,
        export_params=True, 
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True, 
        opset_version=12,
        dynamic_axes = dynamic_axes
    )
    print(f"ONNX model saved to {onnx_path}")

# Function to generate RKNN model
def generate_rknn_model(onnx_path, rknn_path):
    # Create RKNN object
    rknn = RKNN()

    dynamic_input = [
        [[25, 128]],
        [[50, 128]],
        [[75, 128]],
        [[100, 128]],
        [[125, 128]],
        [[150, 128]],
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

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')

    # Release RKNN object for next use
    rknn.release()

if __name__ == "__main__":
    # Paths
    asd_model_path = "weight/pretrain_AVA_CVPR.model"

    # Define the base path for models
    base_model_path = "./models/lossAV"

    # Use the base path to construct the full paths
    scripted_model_path = f"{base_model_path}.pt"
    onnx_model_path = f"{base_model_path}.onnx"
    rknn_model_path = f"{base_model_path}.rknn"
    
    # Steps
    create_torchscript_model(asd_model_path, scripted_model_path)
    export_onnx_model(scripted_model_path, onnx_model_path)
    generate_rknn_model(onnx_model_path, rknn_model_path)
