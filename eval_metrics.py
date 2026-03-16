import argparse
from cleanfid import fid

def main():
    parser = argparse.ArgumentParser(description="Evaluate gFID for LlamaGen + MAVT")
    parser.add_argument("--fake_dir", type=str, required=True, help="Thư mục chứa ảnh do model sinh ra")
    parser.add_argument("--real_dir", type=str, required=True, help="Thư mục chứa ảnh thật (VD: ImageNet/COCO)")
    args = parser.parse_args()

    print(f"Đang tính toán FID...\nFake data: {args.fake_dir}\nReal data: {args.real_dir}")
    
    # Tính FID
    score = fid.compute_fid(args.fake_dir, args.real_dir, mode="clean")
    print(f"=====================================")
    print(f"KẾT QUẢ gFID SCORE: {score:.4f}")
    print(f"=====================================")

if __name__ == "__main__":
    main()