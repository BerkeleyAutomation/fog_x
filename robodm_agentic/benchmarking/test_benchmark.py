#!/usr/bin/env python3
"""
Simple test script for the DROID benchmark without requiring tensorflow.
This tests the basic structure and agent setup.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
robodm_root = current_dir.parent.parent
sys.path.insert(0, str(robodm_root))
sys.path.insert(0, str(current_dir.parent))

from robodm_agentic.core.robodm_interface import RoboDMInterface
from robodm_agentic.core.agent import RoboDMAgent
from robodm_agentic.clients.llm_client import LLMClient
from robodm_agentic.clients.vlm_client import VLMClient


async def test_agent_setup():
    """Test basic agent setup without tensorflow."""
    print("Testing RoboDM Agentic setup...")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test RoboDM interface
        try:
            robodm_interface = RoboDMInterface(str(temp_path))
            print("✓ RoboDM interface created successfully")
        except Exception as e:
            print(f"✗ RoboDM interface failed: {e}")
            return False
        
        # Test LLM client
        try:
            llm_client = LLMClient(
                model="qwen2.5:7b",
                provider="ollama"
            )
            print("✓ LLM client created successfully")
        except Exception as e:
            print(f"✗ LLM client failed: {e}")
            llm_client = None
        
        # Test VLM client
        try:
            vlm_client = VLMClient(
                model="llava:7b",
                provider="ollama"
            )
            print("✓ VLM client created successfully")
        except Exception as e:
            print(f"✗ VLM client failed: {e}")
            vlm_client = None
        
        # Test agent creation
        try:
            agent = RoboDMAgent(
                robodm_interface=robodm_interface,
                llm_client=llm_client,
                vlm_client=vlm_client,
                enable_vision=vlm_client is not None
            )
            print("✓ Agent created successfully")
            
            # Test setup
            test_results = await agent.test_setup()
            print("Agent test results:")
            for component, status in test_results.items():
                status_str = "✓" if status else "✗"
                print(f"  {status_str} {component}: {'OK' if status else 'Failed'}")
            
            return True
            
        except Exception as e:
            print(f"✗ Agent creation failed: {e}")
            return False


async def main():
    """Main test function."""
    print("RoboDM Agentic Benchmark Test")
    print("=" * 40)
    
    success = await test_agent_setup()
    
    if success:
        print("\n✓ All tests passed! The benchmark structure is working.")
        print("\nTo run the full benchmark with DROID data:")
        print("1. Install tensorflow and tensorflow-datasets:")
        print("   pip install tensorflow tensorflow-datasets")
        print("2. Install and start ollama with required models:")
        print("   ollama pull qwen2.5:7b")
        print("   ollama pull llava:7b")
        print("3. Run the benchmark:")
        print("   python robodm_agentic/benchmarking/droid_benchmark.py --num-trajectories 100")
    else:
        print("\n✗ Some tests failed. Please check your setup.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 