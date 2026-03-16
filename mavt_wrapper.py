import torch
import torch.nn as nn

# Import từ thư mục mavt của team bạn
from mavt.tokenizer import MAVTokenizer
from mavt.types import LatentOutput

class MAVTForLlamaGen(nn.Module):
    """
    Lớp vỏ bọc (Wrapper) biến MAVT thành định dạng VQ-VAE mà LlamaGen yêu cầu.
    """
    def __init__(self, device='cuda', latent_dim=32, codebook_size=16384):
        super().__init__()
        self.mavt = MAVTokenizer().to(device)
        self.mavt.eval()
        self.device = device
        
        # BẢN VÁ TẠM THỜI (WORKAROUND): 
        # Map từ LlamaGen discrete indices (B, N) -> MAVT continuous latent (B, N, 32)
        # Báo team cập nhật module BSQ Quantization vào đây sau.
        self.dummy_quantizer = nn.Embedding(codebook_size, latent_dim).to(device)

    @torch.no_grad()
    def decode_code(self, index_sample, qzshape):
        """
        Hàm này thay thế trực tiếp cho vq_model.decode_code() của LlamaGen
        - index_sample: Tensor số nguyên, shape (Batch, Num_Tokens)
        - qzshape: [Batch, embed_dim, latent_size, latent_size]
        """
        B = index_sample.shape[0]
        N = index_sample.shape[1]
        
        # 1. Chuyển đổi ID -> Latent z (B, N, 32)
        z = self.dummy_quantizer(index_sample)
        
        # 2. Đóng gói vào format MAVT yêu cầu
        dummy_understand = torch.zeros(B, 768, device=self.device)
        latent_in = LatentOutput(z=z, z_understand=dummy_understand, mu=z, log_var=z)
        dummy_positions = torch.zeros(B, N, 4, device=self.device)
        
        # 3. Tính toán kích thước ảnh đầu ra
        # LlamaGen tạo ảnh vuông. MAVT dùng patch_size mặc định là 16.
        latent_size = qzshape[2]
        target_size = latent_size * 16 
        target_shape = (B, 3, target_size, target_size)
        
        # 4. Giải mã bằng MAVT
        decoder_out = self.mavt.decode(
            latent_out=latent_in,
            positions=dummy_positions,
            target_shape=target_shape,
            modality="image"
        )
        
        return decoder_out.reconstruction