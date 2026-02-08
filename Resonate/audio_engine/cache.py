"""
Caching System - Performance optimization for audio processing

Caches expensive intermediate results (separated stems, enhanced audio)
to disk to avoid re-processing when tweaking parameters.
"""

import os
import json
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pickle

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry:
    """Metadata for a cached item."""
    key: str
    created_at: str
    file_path: str
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"  # Cache version for invalidation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
            "version": self.version
        }


class CacheManager:
    """
    Manages disk-based caching for audio processing results.
    
    Provides:
    - Stem caching (separated vocals, drums, bass, other)
    - Enhanced audio caching (processed stems)
    - Metadata caching (analysis results, parameters)
    - Automatic cleanup based on size limits
    - Cache validation and integrity checking
    """
    
    def __init__(self, cache_dir: str = None, max_size_gb: float = 10.0,
                 auto_cleanup_percent: float = 0.2):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Base cache directory (defaults to ./cache)
            max_size_gb: Maximum cache size in GB
            auto_cleanup_percent: Fraction to clean when over limit
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "cache"
        
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.auto_cleanup_percent = auto_cleanup_percent
        
        # Create cache structure
        self.stems_dir = self.cache_dir / "stems"
        self.enhanced_dir = self.cache_dir / "enhanced"
        self.metadata_dir = self.cache_dir / "metadata"
        
        # Create directories
        for dir_path in [self.stems_dir, self.enhanced_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load cache index
        self._index_file = self.cache_dir / "cache_index.json"
        self._index = self._load_index()
        
        logger.info(f"Cache initialized: {self.cache_dir}")
        logger.info(f"  Max size: {max_size_gb:.1f} GB")
    
    def _load_index(self) -> Dict[str, CacheEntry]:
        """Load cache index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r') as f:
                    data = json.load(f)
                    return {entry['key']: CacheEntry(**entry) 
                           for entry in data}
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_index(self):
        """Save cache index to disk."""
        try:
            with open(self._index_file, 'w') as f:
                json.dump([entry.to_dict() for entry in self._index.values()], 
                         f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Combine all arguments into a string
        key_parts = []
        
        for arg in args:
            if isinstance(arg, np.ndarray):
                # Hash array content
                hash_obj = hashlib.md5(arg.tobytes())
                key_parts.append(hash_obj.hexdigest()[:16])
            elif isinstance(arg, (str, int, float)):
                key_parts.append(str(arg))
            else:
                # Use repr for other types
                key_parts.append(str(arg)[:32])
        
        # Add keyword arguments
        for k, v in sorted(kwargs.items()):
            if isinstance(v, dict):
                v = json.dumps(v, sort_keys=True)
            key_parts.append(f"{k}:{v}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, category: str, key: str, extension: str = ".npy") -> Path:
        """Get cache file path for a key (ensures parents exist)."""
        if category == "stems":
            dir_path = self.stems_dir / key[:2]
        elif category == "se":
            dir_path = self.stems_dir / "se"
        elif category == "enhanced":
            dir_path = self.enhanced_dir / key[:2]
        else:
            dir_path = self.metadata_dir

        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{key}{extension}"
    
    # === Stem Caching ===
    
    def cache_stems(self, audio_hash: str, stems: Dict[str, np.ndarray],
                   metadata: Dict[str, Any] = None) -> str:
        """
        Cache separated stems.
        
        Args:
            audio_hash: Hash of original audio
            stems: Dictionary of stem name -> audio array
            metadata: Optional metadata to store
            
        Returns:
            Cache key for retrieved stems
        """
        key = f"stems_{audio_hash}"
        cache_path = self._get_cache_path("stems", key, extension=".npz")

        # Store stems as numpy archive
        np.savez_compressed(cache_path, **stems)
        
        # Update index
        file_size = cache_path.stat().st_size
        self._index[key] = CacheEntry(
            key=key,
            created_at=datetime.now().isoformat(),
            file_path=str(cache_path),
            size_bytes=file_size,
            metadata={
                "audio_hash": audio_hash,
                "stems": list(stems.keys()),
                **(metadata or {})
            }
        )
        self._save_index()
        
        logger.info(f"Cached stems: {key[:8]}... ({file_size / 1024:.1f} KB)")
        return key
    
    def get_stems(self, audio_hash: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Retrieve cached stems.
        
        Args:
            audio_hash: Hash of original audio
            
        Returns:
            Dictionary of stem name -> audio array, or None if not cached
        """
        key = f"stems_{audio_hash}"
        
        if key not in self._index:
            return None
        
        try:
            cache_path = Path(self._index[key].file_path)
            if not cache_path.exists():
                del self._index[key]
                return None
            
            data = np.load(cache_path)
            stems = {name: data[name].astype(np.float32) for name in data.files}
            
            logger.debug(f"Retrieved cached stems: {key[:8]}...")
            return stems
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached stems: {e}")
            del self._index[key]
            return None
    
    # === Enhanced Audio Caching ===
    
    def cache_enhanced(self, cache_key: str, enhanced_stems: Dict[str, np.ndarray],
                      processing_params: Dict[str, Any] = None) -> str:
        """
        Cache enhanced stems.
        
        Args:
            cache_key: Original stems cache key
            enhanced_stems: Dictionary of stem name -> enhanced audio
            processing_params: Parameters used for enhancement
            
        Returns:
            Enhanced cache key
        """
        enhanced_key = f"enhanced_{cache_key}"
        cache_path = self._get_cache_path("enhanced", enhanced_key)
        
        np.savez_compressed(cache_path, **enhanced_stems)
        
        file_size = cache_path.stat().st_size
        self._index[enhanced_key] = CacheEntry(
            key=enhanced_key,
            created_at=datetime.now().isoformat(),
            file_path=str(cache_path),
            size_bytes=file_size,
            metadata={
                "original_key": cache_key,
                "processing_params": processing_params
            }
        )
        self._save_index()
        
        logger.info(f"Cached enhanced stems: {enhanced_key[:8]}...")
        return enhanced_key
    
    def get_enhanced(self, cache_key: str, 
                    expected_params: Dict[str, Any] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Retrieve cached enhanced stems.
        
        Args:
            cache_key: Original stems cache key
            expected_params: Expected processing parameters (validates match)
            
        Returns:
            Enhanced stems, or None if not cached/mismatched
        """
        enhanced_key = f"enhanced_{cache_key}"
        
        if enhanced_key not in self._index:
            return None
        
        entry = self._index[enhanced_key]
        
        # Check parameters match
        if expected_params and entry.metadata.get("processing_params") != expected_params:
            logger.debug(f"Cache key {enhanced_key[:8]}... has different params, ignoring")
            return None
        
        try:
            cache_path = Path(entry.file_path)
            if not cache_path.exists():
                del self._index[enhanced_key]
                return None
            
            data = np.load(cache_path)
            enhanced = {name: data[name].astype(np.float32) for name in data.files}
            
            logger.debug(f"Retrieved cached enhanced: {enhanced_key[:8]}...")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached enhanced: {e}")
            del self._index[enhanced_key]
            return None
    
    # === Metadata Caching ===
    
    def cache_metadata(self, key: str, metadata: Dict[str, Any]):
        """Cache metadata dictionary."""
        cache_path = self._get_cache_path("metadata", key, ".json")
        
        with open(cache_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        file_size = cache_path.stat().st_size
        self._index[key] = CacheEntry(
            key=key,
            created_at=datetime.now().isoformat(),
            file_path=str(cache_path),
            size_bytes=file_size,
            metadata=metadata
        )
        self._save_index()
    
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached metadata."""
        if key not in self._index:
            return None
        
        try:
            cache_path = Path(self._index[key].file_path)
            if not cache_path.exists():
                del self._index[key]
                return None
            
            with open(cache_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.warning(f"Failed to retrieve metadata: {e}")
            del self._index[key]
            return None
    
    # === Cache Management ===
    
    def get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total = 0
        for entry in self._index.values():
            path = Path(entry.file_path)
            if path.exists():
                total += path.stat().st_size
        return total
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        size = self.get_cache_size()
        count = len(self._index)
        
        return {
            "total_size_bytes": size,
            "total_size_gb": size / (1024**3),
            "item_count": count,
            "max_size_gb": self.max_size_bytes / (1024**3),
            "usage_percent": (size / self.max_size_bytes) * 100
        }
    
    def cleanup(self, target_size_bytes: int = None):
        """
        Clean up oldest cache entries to free space.
        
        Args:
            target_size_bytes: Target cache size (defaults to max_size * (1 - auto_cleanup_percent))
        """
        if target_size_bytes is None:
            target_size_bytes = int(self.max_size_bytes * (1 - self.auto_cleanup_percent))
        
        current_size = self.get_cache_size()
        
        if current_size <= target_size_bytes:
            logger.debug("Cache cleanup not needed")
            return
        
        # Sort by creation time (oldest first)
        sorted_entries = sorted(
            self._index.values(),
            key=lambda e: e.created_at
        )
        
        freed = 0
        to_delete = []
        
        for entry in sorted_entries:
            if current_size - freed <= target_size_bytes:
                break
            
            to_delete.append(entry.key)
            freed += entry.size_bytes
        
        # Delete entries
        for key in to_delete:
            entry = self._index[key]
            path = Path(entry.file_path)
            if path.exists():
                path.unlink()
            del self._index[key]
            logger.info(f"Cleaned up cache entry: {key[:8]}...")
        
        self._save_index()
        logger.info(f"Cache cleanup: freed {freed / 1024 / 1024:.1f} MB")
    
    def clear_all(self):
        """Clear all cache data."""
        for entry in list(self._index.values()):
            path = Path(entry.file_path)
            if path.exists():
                path.unlink()
        
        self._index.clear()
        self._save_index()
        logger.info("Cache cleared")
    
    def invalidate(self, pattern: str):
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match (e.g., "stems_*")
        """
        import fnmatch
        
        to_delete = []
        for key in self._index:
            if fnmatch.fnmatch(key, pattern):
                to_delete.append(key)
        
        for key in to_delete:
            entry = self._index[key]
            path = Path(entry.file_path)
            if path.exists():
                path.unlink()
            del self._index[key]
            logger.info(f"Invalidated cache entry: {key}")
        
        if to_delete:
            self._save_index()
    
    def invalidate_by_version(self, required_version: str):
        """
        Invalidate cache entries with version older than required.
        
        Args:
            required_version: Minimum version required (e.g., "2.0")
        """
        to_delete = []
        for key, entry in self._index.items():
            try:
                cached_version = entry.version or "1.0"
                if cached_version < required_version:
                    to_delete.append(key)
            except (ValueError, TypeError):
                to_delete.append(key)
        
        for key in to_delete:
            entry = self._index[key]
            path = Path(entry.file_path)
            if path.exists():
                path.unlink()
            del self._index[key]
            logger.info(f"Invalidated outdated cache entry: {key}")
        
        if to_delete:
            self._save_index()
    
    # === Segment Caching ===
    
    def cache_segment(self, base_key: str, segment_index: int,
                     segment_data: np.ndarray, metadata: Dict[str, Any] = None) -> str:
        """Cache a segment of audio (e.g., 30-second chunk)."""
        segment_key = f"seg_{base_key}_{segment_index}"
        cache_path = self._get_cache_path("se", segment_key, extension=".npz")
        
        np.savez_compressed(cache_path, segment=segment_data)
        
        file_size = cache_path.stat().st_size
        self._index[segment_key] = CacheEntry(
            key=segment_key,
            created_at=datetime.now().isoformat(),
            file_path=str(cache_path),
            size_bytes=file_size,
            metadata={"base_key": base_key, "segment_index": segment_index,
                     "segment_length": len(segment_data), **(metadata or {})},
            version="2.0"
        )
        self._save_index()
        
        logger.debug(f"Cached segment {segment_index}: {segment_key[:8]}...")
        return segment_key
    
    def get_segment(self, base_key: str, segment_index: int) -> Optional[np.ndarray]:
        """Retrieve a cached segment."""
        segment_key = f"seg_{base_key}_{segment_index}"
        
        if segment_key not in self._index:
            return None
        
        try:
            cache_path = Path(self._index[segment_key].file_path)
            if not cache_path.exists():
                del self._index[segment_key]
                return None
            
            data = np.load(cache_path)
            return data['segment'].astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Failed to retrieve segment {segment_index}: {e}")
            del self._index[segment_key]
            return None
    
    def cache_segments(self, base_key: str, segments: list,
                      metadata: Dict[str, Any] = None) -> list:
        """Cache multiple segments at once."""
        keys = []
        for i, segment in enumerate(segments):
            keys.append(self.cache_segment(base_key, i, segment, metadata))
        
        logger.info(f"Cached {len(segments)} segments for {base_key[:8]}...")
        return keys
    
    def get_segments(self, base_key: str, num_segments: int) -> Optional[list]:
        """Retrieve all cached segments for a base key."""
        segments = []
        
        for i in range(num_segments):
            segment = self.get_segment(base_key, i)
            if segment is None:
                logger.warning(f"Missing segment {i} for {base_key[:8]}...")
                return None
            segments.append(segment)
        
        return segments
    
    def invalidate_segments(self, base_key: str):
        """Invalidate all segments for a base key."""
        import fnmatch
        pattern = f"seg_{base_key}_*"
        
        to_delete = [key for key in self._index if fnmatch.fnmatch(key, pattern)]
        
        for key in to_delete:
            path = Path(self._index[key].file_path)
            if path.exists():
                path.unlink()
            del self._index[key]
        
        if to_delete:
            self._save_index()
            logger.info(f"Invalidated {len(to_delete)} segments for {base_key[:8]}...")


# Global cache manager instance
_default_cache_manager: Optional[CacheManager] = None


def get_cache_manager(cache_dir: str = None) -> CacheManager:
    """
    Get or create default cache manager.
    
    Args:
        cache_dir: Optional cache directory
        
    Returns:
        CacheManager instance
    """
    global _default_cache_manager
    
    if _default_cache_manager is None:
        _default_cache_manager = CacheManager(cache_dir)
    
    return _default_cache_manager


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create cache manager
    cache = CacheManager(cache_dir="./test_cache", max_size_gb=0.1)
    print(f"Cache created: {cache.cache_dir}")
    
    # Test stem caching
    stems = {
        'vocals': np.random.randn(44100).astype(np.float32),
        'drums': np.random.randn(44100).astype(np.float32),
        'bass': np.random.randn(44100).astype(np.float32),
        'other': np.random.randn(44100).astype(np.float32)
    }
    
    audio_hash = "test_audio_123"
    
    # Cache stems
    cache_key = cache.cache_stems(audio_hash, stems, metadata={"test": True})
    print(f"Cached stems: {cache_key}")
    
    # Retrieve stems
    retrieved = cache.get_stems(audio_hash)
    if retrieved:
        print(f"✅ Retrieved stems: {list(retrieved.keys())}")
    
    # Get cache info
    info = cache.get_cache_info()
    print(f"Cache info: {info}")
    
    # Cleanup
    cache.clear_all()
    
    # Remove test directory
    import shutil
    if Path("./test_cache").exists():
        shutil.rmtree("./test_cache")
