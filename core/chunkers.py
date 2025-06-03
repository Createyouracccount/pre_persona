from abc import ABC, abstractmethod
from typing import List, Dict
import re
from config import RAGConfig

class BaseChunker(ABC):
    """청킹 기본 클래스"""
    
    @abstractmethod
    def chunk(self, content: str, metadata: Dict) -> List[Dict]:
        pass

class TopicTurnChunker(BaseChunker):
    """토픽 중심 턴 단위 청킹"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.topic_change_keywords = [
            '그런데', '아 그리고', '추가로', '또 다른', '혹시',
            '아 맞다', '아 참', '그러고 보니', '아 그래서'
        ]
    
    def chunk(self, content: str, metadata: Dict) -> List[Dict]:
        """토픽 중심 청킹 수행"""
        turns = self._extract_turns(content)
        boundaries = self._identify_topic_boundaries(turns)
        chunks = self._create_chunks(turns, boundaries, metadata)
        return chunks
    
    def _extract_turns(self, content: str) -> List[Dict]:
        """대화 턴 추출"""
        lines = content.split('\n')
        turns = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('상담사:'):
                turns.append({
                    'speaker': '상담사',
                    'text': line[4:].strip(),
                    'role': 'agent'
                })
            elif line.startswith('손님:'):
                turns.append({
                    'speaker': '손님', 
                    'text': line[3:].strip(),
                    'role': 'customer'
                })
        
        return turns
    
    def _identify_topic_boundaries(self, turns: List[Dict]) -> List[int]:
        """토픽 경계 식별"""
        boundaries = [0]
        
        for i in range(1, len(turns)):
            current_turn = turns[i]['text']
            
            # 토픽 변화 키워드 감지
            for keyword in self.topic_change_keywords:
                if keyword in current_turn and len(current_turn) > 10:
                    if i not in boundaries:
                        boundaries.append(i)
                    break
        
        boundaries.append(len(turns))
        return sorted(list(set(boundaries)))
    
    def _create_chunks(self, turns: List[Dict], boundaries: List[int], metadata: Dict) -> List[Dict]:
        """청크 생성"""
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            chunk_turns = turns[start_idx:end_idx]
            if not chunk_turns:
                continue
            
            chunk_text = '\n'.join([f"{turn['speaker']}: {turn['text']}" for turn in chunk_turns])
            
            chunk = {
                'text': chunk_text,
                'chunk_type': 'topic_turn',
                'category': metadata.get('consulting_category', ''),
                'chunk_id': f"{metadata.get('source_id', '')}_tt_{len(chunks)}",
                'turn_count': len(chunk_turns),
                'metadata': metadata
            }
            chunks.append(chunk)
        
        return chunks

class QAPairChunker(BaseChunker):
    """QA 쌍 기반 청킹"""
    
    def chunk(self, content: str, metadata: Dict) -> List[Dict]:
        lines = content.split('\n')
        chunks = []
        
        customer_lines = []
        agent_lines = []
        
        for line in lines:
            if line.startswith('손님:'):
                customer_lines.append(line)
            elif line.startswith('상담사:'):
                agent_lines.append(line)
                
                if customer_lines:
                    chunk_text = '\n'.join(customer_lines + agent_lines)
                    chunk = {
                        'text': chunk_text,
                        'chunk_type': 'qa_pair',
                        'category': metadata.get('consulting_category', ''),
                        'chunk_id': f"{metadata.get('source_id', '')}_qa_{len(chunks)}",
                        'metadata': metadata
                    }
                    chunks.append(chunk)
                    customer_lines = []
                    agent_lines = []
        
        return chunks

# 청킹 팩토리
class ChunkerFactory:
    """청킹 전략 팩토리"""
    
    _chunkers = {
        'topic_turn_based': TopicTurnChunker,
        'qa_pair_based': QAPairChunker,
    }
    
    @classmethod
    def create_chunker(cls, strategy: str, config: RAGConfig) -> BaseChunker:
        if strategy not in cls._chunkers:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        return cls._chunkers[strategy](config)