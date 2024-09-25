from rknn.api import RKNN

def generate_rknn_model(onnx_model_path, rknn_path, quantize=True):
    rknn = RKNN()

    # Note: the dataset must match the max shape, pick proper files from the generated quantization dataset
    dynamic_input = [
        [[1, 100, 13], [1, 25, 112, 112]],
        [[1, 200, 13], [1, 50, 112, 112]],
        [[1, 300, 13], [1, 75, 112, 112]],
        [[1, 400, 13], [1, 100, 112, 112]],
        [[1, 500, 13], [1, 125, 112, 112]],
        [[1, 600, 13], [1, 150, 112, 112]]
    ]

    print('--> Config model')
    rknn.config(target_platform='rk3588', dynamic_input=dynamic_input, quant_img_RGB2BGR=False)
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=onnx_model_path)
    if ret != 0:
        print('Load ONNX model failed!')
        return ret
    print('done')

    print('--> Building model')
    ret = rknn.build(do_quantization=quantize, dataset='./dataset.txt')
    if ret != 0:
        print('Build RKNN model failed!')
        return ret
    print('done')

    print('--> Export RKNN model')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('Export RKNN model failed!')
        return ret
    print('done')

    rknn.release()

    return 0

if __name__ == '__main__':
    # Define the base path for models
    base_model_path = "./models/lightASD"

    # Use the base path to construct the full paths
    scripted_model_path = f"{base_model_path}.pt"
    onnx_model_path = f"{base_model_path}.onnx"
    rknn_model_path = f"{base_model_path}_i8.rknn"
    
    # Steps
    generate_rknn_model(onnx_model_path, rknn_model_path)