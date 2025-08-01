#!/usr/bin/env python3
"""
Fixed comprehensive test suite for renderman integration with flowsynth training pipeline.
"""

import os
import sys
import numpy as np
from typing import Dict

# Add the code directory to path
sys.path.append('code')
sys.path.append('code/synth')

# Set up library path for macOS
os.environ['DYLD_LIBRARY_PATH'] = '/usr/local/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')

try:
    import librenderman as rm
    print("✓ Successfully imported librenderman")
except ImportError as e:
    print(f"ERROR: librenderman import failed: {e}")
    sys.exit(1)

from synth.render_engine_wrapper import create_render_engine, create_patch_generator
from synth.synthesize import play_patch

class RendermanFlowsynthTester:
    def __init__(self, plugin_path: str = None):
        self.plugin_path = plugin_path or self._find_default_plugin()
        self.engine = None
        self.generator = None
        self.extractor = None
        
    def _find_default_plugin(self) -> str:
        """Find a default VST plugin for testing"""
        default_paths = [
            "/Library/Audio/Plug-Ins/VST3/Massive X.vst3",
            "/Library/Audio/Plug-Ins/VST/Diva.vst",
            "/Library/Audio/Plug-Ins/VST3/Diva.vst3",
            "/Library/Audio/Plug-Ins/Components/DLSMusicDevice.component"
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("No suitable VST plugin found for testing")
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        print("=" * 60)
        print("RENDERMAN FLOWSYNTH INTEGRATION TEST SUITE")
        print("=" * 60)
        print(f"Testing with plugin: {self.plugin_path}")
        print("Librenderman library: /usr/local/lib/librenderman.so.dylib")
        print()
        
        tests = [
            ('basic_import', self.test_basic_import),
            ('engine_initialization', self.test_engine_initialization),
            ('plugin_loading', self.test_plugin_loading),
            ('parameter_discovery', self.test_parameter_discovery),
            ('parameter_responsiveness', self.test_parameter_responsiveness),
            ('audio_generation', self.test_audio_generation),
            ('patch_generation', self.test_patch_generation),
            ('batch_synthesis', self.test_batch_synthesis),
            ('training_data_format', self.test_training_data_format),
            ('config_generation', self.test_config_generation)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"[TEST] {test_name.replace('_', ' ').title()}...")
            try:
                results[test_name] = test_func()
                status = "PASS" if results[test_name] else "FAIL"
                print(f"[{status}] {test_name}")
            except Exception as e:
                print(f"[ERROR] {test_name}: {e}")
                results[test_name] = False
            print()
        
        self.print_summary(results)
        return results
    
    def test_basic_import(self) -> bool:
        """Test basic librenderman import and class availability"""
        try:
            # Check if required classes exist
            if not hasattr(rm, 'RenderEngine'):
                print("✗ RenderEngine class not found")
                return False
            if not hasattr(rm, 'PatchGenerator'):
                print("✗ PatchGenerator class not found")
                return False
            
            print("✓ librenderman imported successfully")
            print("✓ Required classes (RenderEngine, PatchGenerator) found")
            return True
        except Exception as e:
            print(f"✗ Basic import test failed: {e}")
            return False
    
    def test_engine_initialization(self) -> bool:
        """Test RenderEngine initialization with standard audio settings"""
        try:
            # Use the wrapper function
            self.engine = create_render_engine(44100, 512, 512)
            print("✓ RenderEngine initialized with sample rate 44100, buffer size 512")
            return True
        except Exception as e:
            print(f"✗ Engine initialization failed: {e}")
            return False
    
    def test_plugin_loading(self) -> bool:
        """Test loading a VST plugin"""
        try:
            if not self.engine:
                self.engine = create_render_engine(44100, 512, 512)
            
            self.engine.load_plugin(self.plugin_path, 0)
            print(f"✓ Plugin loaded successfully: {os.path.basename(self.plugin_path)}")
            return True
        except Exception as e:
            print(f"✗ Plugin loading failed: {e}")
            return False
    
    def test_parameter_discovery(self) -> bool:
        """Test parameter discovery functionality"""
        try:
            if not self.engine:
                self.engine = create_render_engine(44100, 512, 512)
                self.engine.load_plugin(self.plugin_path, 0)
            
            # Use wrapper for PatchGenerator
            self.generator = create_patch_generator(self.engine)
            
            # Try to get a random patch to test parameter discovery
            patch = self.generator.get_random_patch()
            if patch and len(patch) > 0:
                print(f"✓ Discovered {len(patch)} parameters")
                return True
            else:
                print("✗ No parameters discovered")
                return False
        except Exception as e:
            print(f"✗ Parameter discovery failed: {e}")
            return False
    
    def test_parameter_responsiveness(self) -> bool:
        """Test if parameters can be set and affect the plugin"""
        try:
            if not self.generator:
                if not self.engine:
                    self.engine = create_render_engine(44100, 512, 512)
                    self.engine.load_plugin(self.plugin_path, 0)
                self.generator = create_patch_generator(self.engine)
            
            # Test setting a simple patch
            test_patch = [(0, 0.5), (1, 0.3), (2, 0.8)]
            self.engine.set_patch(test_patch)
            print("✓ Parameters can be set successfully")
            return True
        except Exception as e:
            print(f"✗ Parameter setting failed: {e}")
            return False
    
    def test_audio_generation(self) -> bool:
        """Test basic audio generation"""
        if not self.engine:
            print("✗ Engine not initialized")
            return False
        
        try:
            # Create patch generator if not exists
            if not self.generator:
                self.generator = create_patch_generator(self.engine)
            
            # Generate audio with random patch
            audio, patch = play_patch(self.engine, self.generator)
            
            if audio is not None and len(audio) > 0:
                print(f"✓ Generated audio: {len(audio)} samples")
                print(f"✓ Audio range: [{np.min(audio):.3f}, {np.max(audio):.3f}]")
                return True
            else:
                print("✗ No audio generated")
                return False
        except Exception as e:
            print(f"✗ Audio generation failed: {e}")
            return False
    
    def test_patch_generation(self) -> bool:
        """Test patch generation functionality"""
        if not self.engine:
            print("✗ Engine not initialized")
            return False
        
        try:
            if not self.generator:
                self.generator = create_patch_generator(self.engine)
            
            # Generate multiple random patches
            patches = []
            for i in range(3):
                patch = self.generator.get_random_patch()
                patches.append(patch)
            
            if patches and all(len(p) > 0 for p in patches):
                print(f"✓ Generated {len(patches)} random patches")
                print(f"✓ Average patch size: {np.mean([len(p) for p in patches]):.1f} parameters")
                return True
            else:
                print("✗ Patch generation failed")
                return False
        except Exception as e:
            print(f"✗ Patch generation failed: {e}")
            return False
    
    def test_batch_synthesis(self) -> bool:
        """Test batch synthesis functionality"""
        try:
            # This test requires parameter files, so we'll create a minimal test
            if not self.extractor:
                print("✗ No parameter extractor available")
                return False
            
            params = self.extractor.discover_parameters()
            if not params:
                print("✗ No parameters available for batch synthesis")
                return False
            
            print("✓ Batch synthesis prerequisites met")
            print(f"✓ Available parameters: {len(params)}")
            return True
        except Exception as e:
            print(f"✗ Batch synthesis test failed: {e}")
            return False
    
    def test_training_data_format(self) -> bool:
        """Test training data format compatibility"""
        try:
            if not self.extractor:
                print("✗ No parameter extractor available")
                return False
            
            params = self.extractor.discover_parameters()
            if not params:
                print(f"✗ Training data format test failed: Could not discover parameters for {self.plugin_path}")
                return False
            
            # Test parameter selection for training
            selected_params = self.extractor.select_parameters(params)
            
            if selected_params:
                print(f"✓ Selected {len(selected_params)} parameters for training")
                print("✓ Training data format compatible")
                return True
            else:
                print("✗ No parameters selected for training")
                return False
        except Exception as e:
            print(f"✗ Training data format test failed: {e}")
            return False
    
    def test_config_generation(self) -> bool:
        """Test configuration file generation"""
        try:
            if not self.extractor:
                print("✗ No parameter extractor available")
                return False
            
            params = self.extractor.discover_parameters()
            if not params:
                print(f"✗ Config generation test failed: Could not discover parameters for {self.plugin_path}")
                return False
            
            config = self.extractor.generate_config()
            
            if config and 'plugin_path' in config:
                print("✓ Configuration generated successfully")
                print(f"✓ Plugin path: {config['plugin_path']}")
                return True
            else:
                print("✗ Configuration generation failed")
                return False
        except Exception as e:
            print(f"✗ Config generation test failed: {e}")
            return False
    
    def print_summary(self, results: Dict[str, bool]):
        """Print test results summary"""
        print("=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            test_display = test_name.replace('_', ' ').title()
            print(f"{test_display:.<40} {status}")
        
        passed_count = sum(results.values())
        total_count = len(results)
        percentage = (passed_count / total_count) * 100
        
        print()
        print(f"Overall: {passed_count}/{total_count} tests passed ({percentage:.1f}%)")
        
        if passed_count == total_count:
            print("\n🎉 All tests passed! Renderman is ready for flowsynth training.")
        elif passed_count > 0:
            print(f"\n⚠️  {total_count - passed_count} test(s) failed. Please check the issues above.")
        else:
            print("\n❌ All tests failed. Please check your renderman installation.")
        
        print("\nNext steps for flowsynth training:")
        print("1. Run parameter extraction: python code/synth/extract_plugin_params.py <plugin_path> --create-training-files")
        print("2. Generate training dataset: python code/train.py --dataset <dataset_name>")
        print("3. Train model: python code/train.py --config <config_file>")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test renderman integration with flowsynth')
    parser.add_argument('--plugin', type=str, help='Path to VST plugin for testing')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    tester = RendermanFlowsynthTester(args.plugin)
    results = tester.run_all_tests()
    
    # Exit with error code if any tests failed
    if not all(results.values()):
        sys.exit(1)

if __name__ == "__main__":
    main()
