from rknn.api import RKNN

def convert_onnx_to_rknn(onnx_path, rknn_path, do_quantization=False):
    rknn = RKNN(verbose=True)

    # --- Step 1: config ---
    print('--> Configuring RKNN model')
    rknn.config(
        mean_values=[[0, 0, 0]],
        # THIS IS THE CRITICAL CHANGE
        std_values=[[255.0, 255.0, 255.0]],  # Tell NPU to divide R,G,B channels by 255
        target_platform='rk3588',
        optimization_level=3,
        # You can also specify the input format
    )

    # Step 2: load ONNX
    print('--> Loading ONNX model')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('Load ONNX model failed')
        exit(ret)
    print('done')

    # Step 3: build RKNN (with/without quantization)
    print('--> Building RKNN model (quantization: {})'.format(do_quantization))
    dataset = 'dataset.txt' if do_quantization else None
    ret = rknn.build(do_quantization=do_quantization, dataset=dataset)

    if ret != 0:
        print('Build RKNN model failed')
        exit(ret)
    print('done')

    # Step 4: export
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('Export RKNN model failed')
        exit(ret)
    print('done')
    rknn.release()

if __name__ == '__main__':
    onnx_path = 'epoch56_sim.onnx'     # Path to your YOLOv8 ONNX file
    rknn_path = 'epoch56_sim.rknn'  # Output path for the RKNN file
    convert_onnx_to_rknn(onnx_path, rknn_path, do_quantization=True)
