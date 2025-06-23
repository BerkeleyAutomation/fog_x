#!/usr/bin/env python3
"""Test script for RoboDM Agentic Framework installation."""

import os
import sys
from pathlib import Path

# Add necessary paths
current_dir = Path(__file__).parent
robodm_root = current_dir.parent
sys.path.insert(0, str(robodm_root))
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")

    try:
        import robodm
        print("✓ RoboDM core package")
    except ImportError as e:
        print(f"✗ RoboDM core package: {e}")
        return False

    try:
        from core.robodm_interface import RoboDMInterface
        print("✓ RoboDMInterface")
    except ImportError as e:
        print(f"✗ RoboDMInterface: {e}")
        return False

    try:
        from core.agent import RoboDMAgent
        print("✓ RoboDMAgent")
    except ImportError as e:
        print(f"✗ RoboDMAgent: {e}")
        return False

    try:
        from core.code_executor import CodeExecutor
        print("✓ CodeExecutor")
    except ImportError as e:
        print(f"✗ CodeExecutor: {e}")
        return False

    try:
        from clients.llm_client import LLMClient
        print("✓ LLMClient")
    except ImportError as e:
        print(f"✗ LLMClient: {e}")
        return False

    try:
        from clients.vlm_client import VLMClient
        print("✓ VLMClient")
    except ImportError as e:
        print(f"✗ VLMClient: {e}")
        return False

    try:
        from mcp.server import RoboDMMCPServer
        print("✓ RoboDMMCPServer")
    except ImportError as e:
        print(f"✗ RoboDMMCPServer: {e}")
        return False

    return True

def test_dependencies():
    """Test that optional dependencies are available."""
    print("\\nTesting dependencies...")

    deps_available = {}

    try:
        import numpy
        print("✓ NumPy")
        deps_available['numpy'] = True
    except ImportError:
        print("✗ NumPy (required)")
        deps_available['numpy'] = False

    try:
        from PIL import Image
        print("✓ Pillow")
        deps_available['pillow'] = True
    except ImportError:
        print("✗ Pillow (required)")
        deps_available['pillow'] = False

    try:
        import ollama
        print("✓ Ollama client")
        deps_available['ollama'] = True
    except ImportError:
        print("✗ Ollama client (optional)")
        deps_available['ollama'] = False

    try:
        import openai
        print("✓ OpenAI client")
        deps_available['openai'] = True
    except ImportError:
        print("✗ OpenAI client (optional)")
        deps_available['openai'] = False

    return deps_available

def test_basic_functionality():
    """Test basic functionality without external services."""
    print("\\nTesting basic functionality...")

    try:
        # Test creating a mock trajectory file for testing
        import tempfile

        import numpy as np

        # Create a simple test trajectory
        with tempfile.NamedTemporaryFile(suffix='.vla', delete=False) as tmp_file:
            test_trajectory_path = tmp_file.name

        try:
            import robodm

            # Create test data
            test_data = {
                'camera/rgb': [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)],
                'robot/joint_positions': [np.random.rand(7).astype(np.float32) for _ in range(5)],
                'action': [np.random.rand(3).astype(np.float32) for _ in range(5)]
            }

            # Create trajectory
            robodm.Trajectory.from_dict_of_lists(
                data=test_data,
                path=test_trajectory_path,
                video_codec="rawvideo"
            )

            print("✓ Created test trajectory")

            # Test RoboDM interface
            from core.robodm_interface import RoboDMInterface
            interface = RoboDMInterface(test_trajectory_path)

            trajectories = interface.get_all_trajectories()
            print(f"✓ Found {len(trajectories)} trajectories")

            if trajectories:
                traj_id = trajectories[0]
                metadata = interface.get_trajectory_metadata(traj_id)
                print(f"✓ Retrieved metadata for trajectory {traj_id}")

                data = interface.get_trajectory_data(traj_id)
                print(f"✓ Loaded trajectory data with {len(data)} features")

                frames = interface.get_trajectory_frames(traj_id)
                print(f"✓ Extracted {len(frames)} frames")

            # Test code executor
            from core.code_executor import CodeExecutor
            executor = CodeExecutor(interface)

            test_code = """
result = len(robodm.get_all_trajectories())
print(f"Found {result} trajectories")
"""

            import asyncio
            async def test_execution():
                result = await executor.execute(test_code)
                return result

            exec_result = asyncio.run(test_execution())
            if exec_result['success']:
                print("✓ Code execution successful")
            else:
                print(f"✗ Code execution failed: {exec_result['error']}")
                return False

        finally:
            # Clean up test file
            try:
                os.unlink(test_trajectory_path)
            except:
                pass

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_ollama_connection():
    """Test connection to Ollama if available."""
    print("\\nTesting Ollama connection...")

    try:
        import ollama

        # Try to connect to Ollama
        client = ollama.Client()
        models = client.list()

        print("✓ Connected to Ollama")
        print(f"  Available models: {len(models.get('models', []))}")

        # Check for recommended models
        model_names = [model['name'] for model in models.get('models', [])]

        if 'qwen2.5:7b' in model_names:
            print("✓ Qwen 2.5 7B available")
        else:
            print("✗ Qwen 2.5 7B not found (run: ollama pull qwen2.5:7b)")

        if 'llava:7b' in model_names:
            print("✓ LLaVA 7B available")
        else:
            print("✗ LLaVA 7B not found (run: ollama pull llava:7b)")

        return True

    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
        return False

def main():
    """Run all tests."""
    print("RoboDM Agentic Framework - Installation Test")
    print("=" * 50)

    results = {}

    # Test imports
    results['imports'] = test_imports()

    # Test dependencies
    deps = test_dependencies()
    results['dependencies'] = deps['numpy'] and deps['pillow']

    # Test basic functionality
    if results['imports'] and results['dependencies']:
        results['basic_functionality'] = test_basic_functionality()
    else:
        print("\\nSkipping functionality tests due to missing requirements")
        results['basic_functionality'] = False

    # Test Ollama connection
    results['ollama'] = test_ollama_connection()

    # Summary
    print("\\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20} {status}")

    if all(results.values()):
        print("\\n🎉 All tests passed! The framework is ready to use.")
        return 0
    elif results['imports'] and results['dependencies'] and results['basic_functionality']:
        print("\\n⚠️  Core functionality works, but some optional features may be limited.")
        return 0
    else:
        print("\\n❌ Some core tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
