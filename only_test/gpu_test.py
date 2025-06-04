# RTX 3070 Ti 최적화 설정

import torch
from dataclasses import dataclass
from multiprocessing import cpu_count

@dataclass
class RTX3070Config:
    """RTX 3070 Ti 최적화 GPU 설정
    내 GPU의 최적화를 알아보기 위한 과정
    cmd - nvidia-smi 를 통해 GPU 스펙 확인 후 진행
    """
    # GPU 기본 설정
    embedding_model: str = "dragonkue/bge-m3-ko"
    use_gpu: bool = True
    device: str = "cuda"
    
    # RTX 3070 Ti (8GB VRAM) 최적화
    batch_size: int = 64  # 8GB VRAM에 최적화된 배치 크기
    max_sequence_length: int = 512  # 메모리 효율성을 위한 시퀀스 길이 제한
    
    # 병렬 처리 설정
    max_workers: int = min(6, cpu_count())  # CPU 코어에 맞춘 워커 수
    chunk_size: int = 1000
    
    # 메모리 관리
    save_frequency: int = 500  # 더 자주 저장 (메모리 절약)
    enable_mixed_precision: bool = True  # FP16으로 메모리 절약
    
    # 성능 최적화
    pin_memory: bool = True  # CPU-GPU 전송 속도 향상
    num_workers_dataloader: int = 4  # 데이터 로딩 병렬화

def optimize_for_rtx3070():
    """RTX 3070 Ti를 위한 추가 최적화"""
    if torch.cuda.is_available():
        # GPU 메모리 관리 최적화
        torch.cuda.empty_cache()
        
        # 메모리 분할 전략 최적화 (8GB GPU용)
        torch.cuda.set_per_process_memory_fraction(0.9)  # VRAM의 90% 사용
        
        # cuDNN 최적화
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        print("🎯 RTX 3070 Ti 최적화 완료!")
        print(f"   사용 가능한 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"   최적화된 배치 크기: 64")
        print(f"   혼합 정밀도: 활성화")
        
        return True
    return False

# 수정된 GPU 가속 청킹 시스템 (RTX 3070 Ti 최적화)
class RTX3070OptimizedChunker:
    """RTX 3070 Ti 최적화 청킹 시스템"""
    
    def __init__(self):
        self.config = RTX3070Config()
        self.model = None
        
        # RTX 3070 Ti 최적화 적용
        optimize_for_rtx3070()
        
        print("🚀 RTX 3070 Ti 최적화 청킹 시스템 초기화")
        print(f"📊 설정:")
        print(f"   배치 크기: {self.config.batch_size}")
        print(f"   워커 수: {self.config.max_workers}")
        print(f"   저장 빈도: {self.config.save_frequency}")
        print(f"   혼합 정밀도: {self.config.enable_mixed_precision}")
    
    def initialize_model(self):
        """RTX 3070 Ti 최적화 모델 초기화"""
        if self.model is None:
            print("임베딩 모델 로딩 중... (RTX 3070 Ti 최적화)")
            
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.embedding_model)
            
            # GPU로 모델 이동
            self.model = self.model.to(self.config.device)
            
            # 혼합 정밀도 활성화 (메모리 절약)
            if self.config.enable_mixed_precision:
                self.model.half()  # FP16으로 변환
                print("✅ FP16 혼합 정밀도 활성화 (메모리 50% 절약)")
            
            print(f"✅ 모델 로딩 완료 (RTX 3070 Ti 최적화)")
    
    def process_with_memory_management(self, chunks, output_dir="output"):
        """메모리 관리를 강화한 처리"""
        self.initialize_model()
        
        total_chunks = len(chunks)
        embedded_chunks = []
        
        print(f"🚀 RTX 3070 Ti 배치 처리 시작: {total_chunks}개 청크")
        
        from tqdm import tqdm
        
        for i in tqdm(range(0, total_chunks, self.config.batch_size), desc="임베딩 진행"):
            batch_chunks = chunks[i:i+self.config.batch_size]
            
            try:
                # 배치 처리
                texts = [chunk['text'][:self.config.max_sequence_length] for chunk in batch_chunks]
                
                with torch.no_grad():  # 그래디언트 비활성화
                    # 혼합 정밀도 사용
                    with torch.cuda.amp.autocast(enabled=self.config.enable_mixed_precision):
                        embeddings = self.model.encode(
                            texts,
                            batch_size=self.config.batch_size,
                            device=self.config.device,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True  # 정규화로 성능 향상
                        )
                
                # 임베딩 추가
                for j, chunk in enumerate(batch_chunks):
                    chunk['embedding'] = embeddings[j].tolist()
                    embedded_chunks.append(chunk)
                
                # 더 자주 메모리 정리 (RTX 3070 Ti용)
                if (i + self.config.batch_size) % (self.config.batch_size * 4) == 0:
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                
                # 중간 저장
                if len(embedded_chunks) % self.config.save_frequency == 0:
                    self._save_intermediate(embedded_chunks, output_dir, len(embedded_chunks))
                
            except torch.cuda.OutOfMemoryError:
                print("⚠️ GPU 메모리 부족! 배치 크기를 줄입니다.")
                self.config.batch_size = max(16, self.config.batch_size // 2)
                torch.cuda.empty_cache()
                continue
                
            except Exception as e:
                print(f"❌ 배치 처리 오류: {e}")
                # 실패한 배치는 임베딩 없이 추가
                for chunk in batch_chunks:
                    chunk['embedding'] = None
                    embedded_chunks.append(chunk)
        
        # 최종 메모리 정리
        torch.cuda.empty_cache()
        
        return embedded_chunks
    
    def _save_intermediate(self, chunks, output_dir, count):
        """중간 저장"""
        import os
        import pickle
        
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"rtx3070_intermediate_{count}.pkl")
        
        with open(filename, 'wb') as f:
            pickle.dump(chunks, f)
        
        print(f"💾 중간 저장: {count}개 청크 (RTX 3070 Ti 최적화)")

# 성능 벤치마크 함수
def benchmark_rtx3070():
    """RTX 3070 Ti 성능 벤치마크"""
    import time
    import torch
    
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다.")
        return
    
    print("🔍 RTX 3070 Ti 성능 벤치마크 시작...")
    
    # GPU 정보
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"📊 GPU 정보:")
    print(f"   이름: {gpu_props.name}")
    print(f"   VRAM: {gpu_props.total_memory / 1e9:.1f}GB")
    print(f"   SM 개수: {gpu_props.multi_processor_count}")
    
    # 간단한 성능 테스트
    test_sizes = [32, 64, 128, 256]
    
    for batch_size in test_sizes:
        try:
            # 테스트 텐서 생성
            test_tensor = torch.randn(batch_size, 768, device='cuda')
            
            # 시간 측정
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(100):
                    result = torch.matmul(test_tensor, test_tensor.T)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            processing_time = end_time - start_time
            throughput = (batch_size * 100) / processing_time
            
            print(f"   배치 크기 {batch_size:3d}: {throughput:7.1f} samples/sec")
            
        except torch.cuda.OutOfMemoryError:
            print(f"   배치 크기 {batch_size:3d}: 메모리 부족")
            break
        finally:
            torch.cuda.empty_cache()

def run_rtx3070_optimized():
    """RTX 3070 Ti 최적화 실행"""
    
    # 벤치마크 실행
    benchmark_rtx3070()
    
    # 최적화된 청킹 시스템 실행
    chunker = RTX3070OptimizedChunker()
    
    return chunker

if __name__ == "__main__":
    run_rtx3070_optimized()