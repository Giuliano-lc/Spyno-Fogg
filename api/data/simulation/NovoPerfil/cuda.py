import torch

print("CUDA disponível:", torch.cuda.is_available())
print("Qtd GPUs:", torch.cuda.device_count())
print("GPU atual:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma")
print("Versão torch:", torch.__version__)
print("Versão CUDA do torch:", torch.version.cuda)