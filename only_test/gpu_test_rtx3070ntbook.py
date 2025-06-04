# RTX 3070 Ti 벤치마크 기반 최적화 설정

import torch
from dataclasses import dataclass
from multiprocessing import cpu_count

@dataclass
class RTX3070OptimizedConfig:
    """벤치마크 결과 기반 RTX 3070 Ti 최적화 설정"""
    # GPU 기본 설정
    embedding_model: str = "dragonkue/bge-m3-ko"
    use_gpu: bool = True
    device: str = "cuda"
    
    # 벤치마크 기반 최적화
    batch_size: int = 128  # 안정성과 성능의 균형점
    max_batch_size: int = 256  # 최대 성능 배치 크기
    fallback_batch_size: int = 64  # 메모리 부족 시 대안
    
    max_sequence_length: int = 512
    
    # 병렬 처리 설정 (40 SM 활용)
    max_workers: int = min(8, cpu_count())  # SM 개수를 고려한 워커 수 증가
    chunk_size: int = 1500  # 처리량 증가에 맞춰 청크 크기 증가
    
    # 메모리 관리 (8.6GB VRAM 활용)
    save_frequency: int = 750  # 처리량 증가에 맞춰 조정
    enable_mixed_precision: bool = True
    memory_fraction: float = 0.85  # 8.6GB의 85% 사용 (더 보수적)
    
    # 성능 최적화
    pin_memory: bool = True
    num_workers_dataloader: int = 6  # 늘린 워커 수에 맞춰 조정
    
    # 동적 배치 크기 조정
    adaptive_batch_size: bool = True
    performance_target: float = 2000000  # 목표 처리량 (samples/sec)

def dynamic_batch_optimization():
    """동적 배치 크기 최적화"""
    if not torch.cuda.is_available():
        return 32
    
    # 실제 임베딩 모델로 테스트
    test_sizes = [64, 128, 192, 256, 320]
    best_batch_size = 64
    best_throughput = 0
    
    print("🔍 실제 임베딩 모델로 최적 배치 크기 찾는 중...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("dragonkue/bge-m3-ko")
        model = model.to("cuda")
        model.half()  # FP16으로 메모리 절약
        
        # 테스트 텍스트 준비
        test_texts = ["테스트 문장입니다." * 10] * 320  # 최대 배치 크기만큼
        
        import time
        
        for batch_size in test_sizes:
            try:
                # 3번 측정 후 평균
                times = []
                for _ in range(3):
                    batch_texts = test_texts[:batch_size]
                    
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    with torch.no_grad():
                        embeddings = model.encode(
                            batch_texts,
                            batch_size=batch_size,
                            device="cuda",
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )
                    
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    torch.cuda.empty_cache()
                
                avg_time = sum(times) / len(times)
                throughput = batch_size / avg_time
                
                print(f"   배치 크기 {batch_size:3d}: {throughput:8.1f} embeddings/sec")
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
                
            except torch.cuda.OutOfMemoryError:
                print(f"   배치 크기 {batch_size:3d}: 메모리 부족")
                break
            except Exception as e:
                print(f"   배치 크기 {batch_size:3d}: 오류 - {e}")
                continue
        
        print(f"✅ 최적 배치 크기: {best_batch_size} (처리량: {best_throughput:.1f} embeddings/sec)")
        return best_batch_size
        
    except Exception as e:
        print(f"❌ 동적 최적화 실패: {e}")
        return 128  # 기본값 반환

class AdaptiveRTX3070Chunker:
    """적응형 RTX 3070 Ti 청킹 시스템"""
    
    def __init__(self):
        self.config = RTX3070OptimizedConfig()
        self.model = None
        self.current_batch_size = self.config.batch_size
        
        # GPU 메모리 최적화
        self._optimize_gpu_memory()
        
        # 동적 배치 크기 최적화
        if self.config.adaptive_batch_size:
            optimal_batch_size = dynamic_batch_optimization()
            self.current_batch_size = optimal_batch_size
            self.config.batch_size = optimal_batch_size
        
        print(f"🚀 적응형 RTX 3070 Ti 청킹 시스템 초기화")
        print(f"📊 최적화된 설정:")
        print(f"   최적 배치 크기: {self.current_batch_size}")
        print(f"   워커 수: {self.config.max_workers}")
        print(f"   청크 크기: {self.config.chunk_size}")
        print(f"   메모리 할당: {self.config.memory_fraction*100:.1f}%")
    
    def _optimize_gpu_memory(self):
        """GPU 메모리 최적화"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            gpu_props = torch.cuda.get_device_properties(0)
            total_vram = gpu_props.total_memory / 1e9
            allocated_vram = total_vram * self.config.memory_fraction
            
            print(f"🎯 GPU 메모리 최적화 완료!")
            print(f"   총 VRAM: {total_vram:.1f}GB")
            print(f"   할당 VRAM: {allocated_vram:.1f}GB")
    
    def process_with_adaptive_batching(self, chunks, output_dir="output"):
        """적응형 배치 처리"""
        self.initialize_model()
        
        total_chunks = len(chunks)
        embedded_chunks = []
        current_batch_size = self.current_batch_size
        
        print(f"🚀 적응형 배치 처리 시작: {total_chunks}개 청크")
        print(f"   초기 배치 크기: {current_batch_size}")
        
        from tqdm import tqdm
        import time
        
        processing_times = []
        
        for i in tqdm(range(0, total_chunks, current_batch_size), desc="임베딩 진행"):
            batch_chunks = chunks[i:i+current_batch_size]
            
            try:
                start_time = time.time()
                
                # 배치 처리
                texts = [chunk['text'][:self.config.max_sequence_length] for chunk in batch_chunks]
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.config.enable_mixed_precision):
                        embeddings = self.model.encode(
                            texts,
                            batch_size=current_batch_size,
                            device=self.config.device,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                
                # 처리 시간 기록
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # 임베딩 추가
                for j, chunk in enumerate(batch_chunks):
                    chunk['embedding'] = embeddings[j].tolist()
                    embedded_chunks.append(chunk)
                
                # 성능 기반 배치 크기 조정
                if len(processing_times) >= 5:  # 5번마다 성능 체크
                    avg_time = sum(processing_times[-5:]) / 5
                    current_throughput = current_batch_size / avg_time
                    
                    # 처리량이 목표보다 낮으면 배치 크기 증가 시도
                    if current_throughput < self.config.performance_target and current_batch_size < self.config.max_batch_size:
                        current_batch_size = min(current_batch_size + 32, self.config.max_batch_size)
                        print(f"📈 배치 크기 증가: {current_batch_size}")
                
                # 메모리 정리
                if (i + current_batch_size) % (current_batch_size * 3) == 0:
                    torch.cuda.empty_cache()
                
                # 중간 저장
                if len(embedded_chunks) % self.config.save_frequency == 0:
                    self._save_intermediate(embedded_chunks, output_dir, len(embedded_chunks))
                
            except torch.cuda.OutOfMemoryError:
                print(f"⚠️ GPU 메모리 부족! 배치 크기 {current_batch_size} → {self.config.fallback_batch_size}")
                current_batch_size = self.config.fallback_batch_size
                torch.cuda.empty_cache()
                continue
                
            except Exception as e:
                print(f"❌ 배치 처리 오류: {e}")
                for chunk in batch_chunks:
                    chunk['embedding'] = None
                    embedded_chunks.append(chunk)
        
        # 성능 통계 출력
        if processing_times:
            avg_processing_time = sum(processing_times) / len(processing_times)
            avg_throughput = self.current_batch_size / avg_processing_time
            
            print(f"\n📊 처리 성능 통계:")
            print(f"   평균 배치 처리 시간: {avg_processing_time:.3f}초")
            print(f"   평균 처리량: {avg_throughput:.1f} embeddings/sec")
            print(f"   최종 배치 크기: {current_batch_size}")
        
        torch.cuda.empty_cache()
        return embedded_chunks
    
    def initialize_model(self):
        """최적화된 모델 초기화"""
        if self.model is None:
            print("🧠 임베딩 모델 로딩 중... (적응형 최적화)")
            
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.embedding_model)
            self.model = self.model.to(self.config.device)
            
            if self.config.enable_mixed_precision:
                self.model.half()
                print("✅ FP16 혼합 정밀도 활성화")
            
            print(f"✅ 모델 로딩 완료")
    
    def _save_intermediate(self, chunks, output_dir, count):
        """중간 저장"""
        import os
        import pickle
        
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"adaptive_rtx3070_{count}.pkl")
        
        with open(filename, 'wb') as f:
            pickle.dump(chunks, f)
        
        print(f"💾 중간 저장: {count}개 청크")

# 사용 예시
def run_adaptive_rtx3070():
    """적응형 RTX 3070 Ti 최적화 실행"""
    
    print("🚀 적응형 RTX 3070 Ti 최적화 시스템")
    print("   - 실시간 성능 모니터링")
    print("   - 동적 배치 크기 조정") 
    print("   - 메모리 사용량 최적화")
    print("   - 벤치마크 기반 설정\n")
    
    chunker = AdaptiveRTX3070Chunker()
    
    print("\n🚀 실제 데이터 처리를 시작하시겠습니까?")
    print("📁 JSON 파일 폴더 경로를 입력해주세요:")
    
    return chunker

if __name__ == "__main__":
    run_adaptive_rtx3070()