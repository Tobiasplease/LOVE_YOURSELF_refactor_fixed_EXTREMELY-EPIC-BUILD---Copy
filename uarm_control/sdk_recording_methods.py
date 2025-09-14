#!/usr/bin/env python3
"""
Investigate official uArm SDK recording methods
"""

from uarm_controller import UarmController


def investigate_sdk_recording_methods():
    """Find the official SDK recording methods"""
    print("Investigating Official uArm SDK Recording Methods")
    print("=================================================")

    controller = UarmController(auto_home=False)

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return

    robot = controller.robot
    print("✅ Connected to uArm")

    # Get all available methods
    all_methods = [method for method in dir(robot) if not method.startswith('_')]

    # Look for recording/playback related methods
    recording_methods = [m for m in all_methods if any(keyword in m.lower() for keyword in
                        ['record', 'play', 'motion', 'move', 'path', 'trajectory', 'sequence'])]

    print(f"\n📋 Found {len(recording_methods)} potentially relevant methods:")
    for method in sorted(recording_methods):
        print(f"  {method}")

    # Test specific recording methods
    print(f"\n🔍 Testing Recording Methods:")

    recording_test_methods = [
        'start_record',
        'stop_record',
        'record_motion',
        'record_position',
        'start_recording',
        'stop_recording',
        'save_motion',
        'load_motion',
        'playback',
        'play_motion',
        'record_path',
        'get_motion_list',
        'motion_record',
        'motion_play'
    ]

    available_recording_methods = []

    for method_name in recording_test_methods:
        if hasattr(robot, method_name):
            available_recording_methods.append(method_name)
            method = getattr(robot, method_name)
            print(f"  ✅ {method_name}: Available")

            # Try to get method signature/help
            try:
                print(f"     Method: {method}")
                # Try to get docstring
                if hasattr(method, '__doc__') and method.__doc__:
                    doc_lines = method.__doc__.strip().split('\n')
                    print(f"     Doc: {doc_lines[0]}")
            except:
                pass
        else:
            print(f"  ❌ {method_name}: Not available")

    # Check for built-in motion recording
    print(f"\n🎬 Testing Built-in Motion Recording:")

    try:
        # Test if we can list existing motions
        if hasattr(robot, 'get_motion_list'):
            motions = robot.get_motion_list()
            print(f"  Existing motions: {motions}")

        # Test playback method we know exists
        if hasattr(robot, 'playback'):
            print(f"  ✅ playback method available")

            # Try to understand playback parameters
            try:
                # This might fail but will show us the expected parameters
                result = robot.playback()
                print(f"  playback() result: {result}")
            except Exception as e:
                print(f"  playback() signature info: {e}")

    except Exception as e:
        print(f"  Error testing built-in methods: {e}")

    # Look for encoder/position streaming methods
    print(f"\n🔧 Position Streaming Methods:")

    streaming_methods = [m for m in all_methods if any(keyword in m.lower() for keyword in
                        ['stream', 'continuous', 'real_time', 'live', 'encoder'])]

    if streaming_methods:
        print(f"  Found streaming methods:")
        for method in streaming_methods:
            print(f"    {method}")
    else:
        print(f"  No obvious streaming methods found")

    controller.disconnect()
    return available_recording_methods


def test_official_recording_workflow():
    """Test the official recording workflow if available"""
    print(f"\n=== Testing Official Recording Workflow ===")

    controller = UarmController(auto_home=False)

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return

    robot = controller.robot

    # Try the workflow we know exists from your previous use
    print("Testing known workflow...")

    try:
        # We know playback exists, so there should be a recording method
        # Let's try some common recording workflows

        # Method 1: Direct recording to named motion
        print("Method 1: Testing direct motion recording...")
        try:
            # This is likely how it works based on your description
            result = robot.start_record('test_motion')
            print(f"start_record result: {result}")
        except AttributeError:
            print("start_record not available")
        except Exception as e:
            print(f"start_record error: {e}")

        # Method 2: Check if recording uses the same system as playback
        print("Method 2: Testing playback-based recording...")
        try:
            # Since playback(motion_name) works, recording might save to motion_name
            result = robot.playback('test_motion')  # This should fail if motion doesn't exist
            print(f"playback test result: {result}")
        except Exception as e:
            print(f"playback error (expected): {e}")

    except Exception as e:
        print(f"Recording workflow test error: {e}")

    controller.disconnect()


def main():
    print("uArm SDK Recording Investigation")
    print("===============================\n")

    # Investigate available methods
    available_methods = investigate_sdk_recording_methods()

    # Test official workflow
    test_official_recording_workflow()

    print(f"\n" + "="*60)
    print("OFFICIAL SDK RECORDING SUMMARY")
    print("="*60)

    if available_methods:
        print(f"✅ Found {len(available_methods)} recording-related methods")
        print("Recommend using official SDK methods instead of custom recording")
    else:
        print("❌ No obvious recording methods found")
        print("May need to check uArm documentation or use different approach")

    print(f"\n📚 Next Steps:")
    print("1. Check uArm Swift Pro documentation for recording API")
    print("2. Use official recording methods if available")
    print("3. If not available, improve custom recording with higher frequency")


if __name__ == "__main__":
    main()