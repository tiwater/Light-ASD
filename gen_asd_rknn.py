import torch
from ASD_2D import ASD
from rknn.api import RKNN

# Function to create TorchScript model
def create_torchscript_model(asd_model_path, save_path):
    # Load pretrained ASD model
    asd_model = ASD()
    asd_model.loadParameters(asd_model_path)
    asd_model.model.eval()
    
    # Script and save the model
    scripted_model = torch.jit.script(asd_model.model)
    torch.jit.save(scripted_model, save_path)
    print(f"TorchScript model saved to {save_path}")

# Function to export ONNX model from TorchScript model
def export_onnx_model(script_model_path, onnx_path):
    # Load the TorchScript model
    torch_script_model = torch.jit.load(script_model_path)
    
    # Example inputs
    dummy_input_audio = torch.randn(1, 100, 13).float()
    dummy_input_visual = torch.randn(1, 25, 112, 112).float()
    
    # Dynamic axes configuration
    dynamic_axes = {
        'audio': {0: 'batch_size', 1: 'time_steps'},
        'visual': {0: 'batch_size', 1: 'frame_sequence'},
        'output': {0: 'batch_size'}
    }
    
    # Export to ONNX
    torch.onnx.export(
        torch_script_model,
        (dummy_input_audio, dummy_input_visual),
        onnx_path,
        input_names=["audio", "visual"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=dynamic_axes
    )
    print(f"ONNX model saved to {onnx_path}")

# Function to export ONNX model from TorchScript model
def export_onnx_static_model(script_model_path, onnx_path):
    # Load the TorchScript model
    torch_script_model = torch.jit.load(script_model_path)
    
    # Example inputs
    dummy_input_audio = torch.randn(1, 100, 13).float()
    dummy_input_visual = torch.randn(1, 25, 112, 112).float()
    
    # Dynamic axes configuration
    dynamic_axes = {
        'audio': {0: 'batch_size', 1: 'time_steps'},
        'visual': {0: 'batch_size', 1: 'frame_sequence'},
        'output': {0: 'batch_size'}
    }
    
    # Export to ONNX
    torch.onnx.export(
        torch_script_model,
        (dummy_input_audio, dummy_input_visual),
        onnx_path,
        input_names=["audio", "visual"],
        output_names=["output"],
        opset_version=17,
        # dynamic_axes=dynamic_axes
    )
    print(f"ONNX model saved to {onnx_path}")

# Function to generate RKNN model
def generate_rknn_model(onnx_path, rknn_path):
    # Create RKNN object
    rknn = RKNN()

    dynamic_input = [
        [[1, 100, 13], [1, 25, 112, 112]],
        [[1, 200, 13], [1, 50, 112, 112]],
        [[1, 300, 13], [1, 75, 112, 112]],
        [[1, 400, 13], [1, 100, 112, 112]],
        [[1, 500, 13], [1, 125, 112, 112]],
        [[1, 600, 13], [1, 150, 112, 112]]
    ]
    
    # Pre-process config
    print('--> Configuring model')
    rknn.config(target_platform='rk3588', optimization_level=3, dynamic_input=dynamic_input)
    print('done')
    
    # Load ONNX model
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
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')
    
    # Release RKNN object
    rknn.release()

# Function to generate RKNN model
def generate_rknn_static_model(onnx_path, rknn_path):
    # Create RKNN object
    rknn = RKNN()
    
    # Pre-process config
    print('--> Configuring model')
    rknn.config(target_platform='rk3588', optimization_level=3)
    print('done')
    
    # Load ONNX model
    print('--> Loading ONNX model')
    ret = rknn.load_onnx(model=onnx_path, input_size_list=[[1, 100, 13], [1, 25, 112, 112]])
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
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')
    
    # Release RKNN object
    rknn.release()

if __name__ == "__main__":
    # Paths
    asd_model_path = "weight/rknn_v21.model"

    # Define the base path for models
    base_model_path = "./models/lightASD"

    # Use the base path to construct the full paths
    scripted_model_path = f"{base_model_path}.pt"
    onnx_model_path = f"{base_model_path}.onnx"
    rknn_model_path = f"{base_model_path}.rknn"
    
    # Steps
    create_torchscript_model(asd_model_path, scripted_model_path)
    export_onnx_model(scripted_model_path, onnx_model_path)
    # Uncomment the following line to generate the RKNN model
    generate_rknn_model(onnx_model_path, rknn_model_path)
