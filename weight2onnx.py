import torch
from resnet50b import *
torch_model = resnet50(num_fc0=6,num_fc1=43,num_fc2=8)#.cuda()
torch_model.load_state_dict(torch.load("diyierpi_model_best.pth.tar")['state_dict'])
batch_size = 1 
input_shape = (3, 448, 448)

# set the model to inference mode
torch_model.eval()

x = torch.randn(batch_size,*input_shape) 	
export_onnx_file = "diyierpi_branch_pt.onnx" 				
torch.onnx.export(
                    torch_model,
                    x,
                    export_onnx_file,
                    input_names=["input"],
                    output_names=["output"],
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
                )

