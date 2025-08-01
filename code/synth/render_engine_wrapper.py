import os
import sys

# Setup macOS library paths before importing librenderman
if sys.platform == 'darwin':
    try:
        from macos_compat import setup_macos_library_paths
        setup_macos_library_paths()
    except ImportError:
        # Fallback for when macos_compat is not in path
        pass

try:
    import librenderman as rm
except ImportError:
    print("Warning: librenderman not available - some functionality will be limited")
    rm = None

import warnings
import gc

def create_render_engine(sample_rate=44100, buffer_size=512, max_buffer_size=512):
    """Create RenderEngine with enhanced memory safety"""
    if rm is None:
        raise ImportError("librenderman is not available")
        
    try:
        # Force garbage collection before creating objects
        gc.collect()
        
        # Method 1: Manual instantiation with memory safety
        engine = rm.RenderEngine.__new__(rm.RenderEngine)
        rm.RenderEngine.__init__(engine, sample_rate, buffer_size, max_buffer_size)
        
        # Verify the engine is properly initialized
        if engine is None:
            raise RuntimeError("Engine initialization returned None")
            
        return engine
    except Exception as e1:
        try:
            gc.collect()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                engine = rm.RenderEngine(sample_rate, buffer_size, max_buffer_size)
                if engine is None:
                    raise RuntimeError("Engine initialization returned None")
                return engine
        except Exception as e2:
            raise RuntimeError(f"All RenderEngine initialization methods failed: {e1}, {e2}")

def create_patch_generator(engine):
    """Create PatchGenerator with enhanced memory safety"""
    if rm is None:
        raise ImportError("librenderman is not available")
        
    if engine is None:
        raise ValueError("Cannot create PatchGenerator with None engine")
        
    try:
        gc.collect()
        
        # Method 1: Manual instantiation with validation
        generator = rm.PatchGenerator.__new__(rm.PatchGenerator)
        rm.PatchGenerator.__init__(generator, engine)
        
        if generator is None:
            raise RuntimeError("PatchGenerator initialization returned None")
            
        return generator
    except Exception as e1:
        try:
            gc.collect()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                generator = rm.PatchGenerator(engine)
                if generator is None:
                    raise RuntimeError("PatchGenerator initialization returned None")
                return generator
        except Exception as e2:
            raise RuntimeError(f"All PatchGenerator initialization methods failed: {e1}, {e2}")